"""
UniRig In-Process Model Cache

Keeps ML models loaded in GPU memory for fast inference.
No separate server process needed - models live in the main ComfyUI process.
"""

import logging
import os
import sys
import inspect
from pathlib import Path
import torch
import lightning as L
import yaml
from box import Box
import numpy as np
import comfy.utils
import comfy.model_management

log = logging.getLogger("unirig")

# Path to vendored unirig (needed for os.chdir in inference)
UNIRIG_PATH = Path(__file__).parent.resolve() / "unirig"

from .unirig.src.inference.download import download
from .unirig.src.data.extract import get_files
from .unirig.src.data.dataset import UniRigDatasetModule, DatasetConfig
from .unirig.src.data.datapath import Datapath
from .unirig.src.data.transform import TransformConfig
from .unirig.src.tokenizer.spec import TokenizerConfig
from .unirig.src.tokenizer.parse import get_tokenizer
from .unirig.src.model.parse import get_model
from .unirig.src.system.parse import get_system, get_writer

# Global model cache - keeps models in GPU memory
_LOADED_MODELS = {}


def load_yaml_config(path: str) -> Box:
    """Load a YAML config file."""
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    return Box(yaml.safe_load(open(path, 'r')))


def load_model_into_memory(model_type: str, task_config_path: str, cache_to_gpu: bool = True):
    """
    Load a model into GPU memory and cache it.

    Args:
        model_type: "skeleton" or "skinning"
        task_config_path: Path to task config YAML
        cache_to_gpu: If True, keep model on GPU; if False, keep on CPU

    Returns:
        Cache key for the loaded model
    """
    cache_key = f"{model_type}_{task_config_path}"

    if cache_key in _LOADED_MODELS:
        log.info("Model %s already loaded", model_type)
        return cache_key

    log.info("Loading %s model...", model_type)

    # Change to UniRig directory for relative paths
    original_cwd = os.getcwd()
    os.chdir(str(UNIRIG_PATH))

    try:
        task = load_yaml_config(task_config_path)

        # Load tokenizer config
        tokenizer_config = task.components.get('tokenizer', None)
        if tokenizer_config is not None:
            tokenizer_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/tokenizer', task.components.tokenizer))
            tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)

        # Build model + system on meta device (zero memory, no random init)
        with torch.device("meta"):
            # Load model config
            model_config_name = task.components.get('model', None)
            if model_config_name is not None:
                model_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/model', model_config_name))
                if tokenizer_config is not None:
                    tokenizer = get_tokenizer(config=tokenizer_config)
                else:
                    tokenizer = None
                model = get_model(tokenizer=tokenizer, **model_config)
            else:
                model = None

        # Load checkpoint
        resume_from_checkpoint = task.get('resume_from_checkpoint', None)
        checkpoint_path = download(resume_from_checkpoint)

        # Build system on meta device
        with torch.device("meta"):
            system_config_name = task.components.get('system', None)
            if system_config_name is not None:
                system_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/system', system_config_name))
                system = get_system(
                    **system_config,
                    model=model,
                    optimizer_config=None,
                    loss_config=None,
                    scheduler_config=None,
                    steps_per_epoch=1,
                )
            else:
                system = None

        # Load checkpoint weights into system (assign=True for meta device)
        if checkpoint_path and system is not None:
            log.info("Loading checkpoint weights...")

            log.info("Loading checkpoint: %s", checkpoint_path)
            state_dict = comfy.utils.load_torch_file(str(checkpoint_path))
            system.load_state_dict(state_dict, strict=False, assign=True)

            device = comfy.model_management.get_torch_device()
            if cache_to_gpu:
                system = system.to(device)
                log.info("Model moved to %s", device)
            else:
                log.info("Model on CPU")

            system.eval()

        # Cache the model components
        _LOADED_MODELS[cache_key] = {
            "task": task,
            "model": model,
            "system": system,
            "tokenizer_config": tokenizer_config,
            "checkpoint_path": checkpoint_path,
            "cache_to_gpu": cache_to_gpu,
        }

        log.info("Model %s loaded and cached", model_type)
        return cache_key

    finally:
        os.chdir(original_cwd)


def apply_config_overrides(config, overrides: dict):
    """
    Apply config overrides to a Box/dict config object.

    Args:
        config: Box or dict config object
        overrides: Dictionary of override values

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    # Apply overrides to transform config (sampler and vertex group)
    if hasattr(config, 'predict_transform_config') or 'predict_transform_config' in config:
        ptc = config.get('predict_transform_config', {})

        # Override sampler config
        if 'sampler_config' in ptc:
            if 'num_samples' in overrides:
                ptc['sampler_config']['num_samples'] = overrides['num_samples']
            if 'vertex_samples' in overrides:
                ptc['sampler_config']['vertex_samples'] = overrides['vertex_samples']

        # Override vertex group config (voxel_skin)
        if 'vertex_group_config' in ptc and 'kwargs' in ptc['vertex_group_config']:
            vg_kwargs = ptc['vertex_group_config']['kwargs']
            if 'voxel_skin' in vg_kwargs:
                vs = vg_kwargs['voxel_skin']
                if 'voxel_grid_size' in overrides:
                    vs['grid'] = overrides['voxel_grid_size']
                # Map voxel_mask_power to alpha (voxel_mask_power is the UI name)
                if 'voxel_mask_power' in overrides:
                    vs['alpha'] = overrides['voxel_mask_power']
                elif 'alpha' in overrides:
                    vs['alpha'] = overrides['alpha']
                if 'grid_query' in overrides:
                    vs['grid_query'] = overrides['grid_query']
                if 'vertex_query' in overrides:
                    vs['vertex_query'] = overrides['vertex_query']
                if 'grid_weight' in overrides:
                    vs['grid_weight'] = overrides['grid_weight']

    return config


def run_inference(cache_key: str, request_data: dict) -> dict:
    """
    Run inference using a cached model.

    Args:
        cache_key: Key for cached model
        request_data: Dictionary containing inference parameters

    Returns:
        Dictionary with inference results
    """
    if cache_key not in _LOADED_MODELS:
        return {"error": f"Model {cache_key} not loaded"}

    cached = _LOADED_MODELS[cache_key]
    task = cached["task"]
    system = cached["system"]
    tokenizer_config = cached["tokenizer_config"]

    # Extract request parameters
    seed = request_data.get("seed", 123)
    input_file = request_data.get("input")
    output_file = request_data.get("output")
    npz_dir = request_data.get("npz_dir")
    cls = request_data.get("cls")
    data_name = request_data.get("data_name")
    config_overrides = request_data.get("config_overrides", {})

    if not all([input_file, output_file, npz_dir]):
        return {"error": "Missing required parameters: input, output, npz_dir"}

    # Change to UniRig directory for relative paths
    original_cwd = os.getcwd()
    os.chdir(str(UNIRIG_PATH))

    try:
        # Set seed
        L.seed_everything(seed, workers=True)

        # Prepare data
        files = get_files(
            data_name=task.components.data_name,
            inputs=input_file,
            input_dataset_dir=None,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        datapath = Datapath(files=files, cls=cls)

        # Load configs
        data_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/data', task.components.data))
        transform_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/transform', task.components.transform))

        # Apply config overrides to transform config
        if config_overrides:
            log.info("Applying config overrides: %s", config_overrides)
            transform_config = apply_config_overrides(transform_config, config_overrides)

        # Get data name
        data_name_actual = task.components.get('data_name', 'raw_data.npz')
        if data_name is not None:
            data_name_actual = data_name

        # Get predict dataset config
        predict_dataset_config = data_config.get('predict_dataset_config', None)
        if predict_dataset_config is not None:
            predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()

        # Get predict transform config
        predict_transform_config = transform_config.get('predict_transform_config', None)
        if predict_transform_config is not None:
            predict_transform_config = TransformConfig.parse(config=predict_transform_config)

        # Create data module
        data = UniRigDatasetModule(
            process_fn=None if system is None or system.model is None else system.model._process_fn,
            train_dataset_config=None,
            predict_dataset_config=predict_dataset_config,
            predict_transform_config=predict_transform_config,
            validate_dataset_config=None,
            train_transform_config=None,
            validate_transform_config=None,
            tokenizer_config=tokenizer_config,
            debug=False,
            data_name=data_name_actual,
            datapath=datapath,
            cls=cls,
        )

        # Setup callbacks
        callbacks = []
        writer_config = task.get('writer', None)
        if writer_config is not None:
            if output_file.endswith('.fbx'):
                writer_config['npz_dir'] = npz_dir
                writer_config['output_dir'] = None
                writer_config['output_name'] = output_file
                # For skeleton inference, disable user_mode to allow NPZ export with bone names
                # This ensures VRoid/template bone names are saved to predict_skeleton.npz
                is_skeleton_inference = 'skeleton' in cache_key.lower()
                writer_config['user_mode'] = not is_skeleton_inference
            callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))

        # Ensure model stays on GPU before inference
        if system is not None and cached.get("cache_to_gpu", True):
            current_device = next(system.parameters()).device
            target_device = comfy.model_management.get_torch_device()
            if current_device.type == 'cpu' and target_device.type != 'cpu':
                system.to(target_device)
                log.info("Model moved back to %s", target_device)

        # Ensure model is in eval mode
        if system is not None:
            system.eval()

        # Override skeleton template if specified
        if cls is not None and system is not None and hasattr(system, 'generate_kwargs'):
            system.generate_kwargs['assign_cls'] = cls
        elif system is not None and hasattr(system, 'generate_kwargs'):
            system.generate_kwargs.pop('assign_cls', None)

        # Create trainer
        trainer_config = task.get('trainer', {})
        target_device = comfy.model_management.get_torch_device()
        if cached.get("cache_to_gpu", True) and target_device.type != 'cpu':
            trainer_config['accelerator'] = 'gpu'
            trainer_config['devices'] = 1

        trainer = L.Trainer(
            callbacks=callbacks,
            logger=None,
            enable_progress_bar=False,
            **trainer_config,
        )

        # Run prediction
        checkpoint_path = cached.get("checkpoint_path")
        # Check if weights_only parameter is supported (Lightning >= 2.6.0)
        predict_sig = inspect.signature(trainer.predict)
        if 'weights_only' in predict_sig.parameters:
            # PyTorch 2.6+ requires weights_only=False for checkpoints with Box objects
            trainer.predict(system, datamodule=data, ckpt_path=checkpoint_path,
                            return_predictions=False, weights_only=False)
        else:
            # Older Lightning versions don't have weights_only parameter
            trainer.predict(system, datamodule=data, ckpt_path=checkpoint_path,
                            return_predictions=False)

        # Keep model on GPU after prediction
        target_device = comfy.model_management.get_torch_device()
        if system is not None and cached.get("cache_to_gpu", True) and target_device.type != 'cpu':
            if next(system.parameters()).device.type == 'cpu':
                system.to(target_device)

        return {"success": True, "output": output_file}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        os.chdir(original_cwd)


def unload_model(cache_key: str):
    """Unload a model from cache to free GPU memory."""
    if cache_key in _LOADED_MODELS:
        cached = _LOADED_MODELS[cache_key]
        # Move model to CPU and delete
        if cached.get("system") is not None:
            cached["system"].cpu()
        del _LOADED_MODELS[cache_key]
        comfy.model_management.soft_empty_cache()
        log.info("Model %s unloaded", cache_key)
        return True
    return False


def list_loaded_models():
    """List all loaded models."""
    return list(_LOADED_MODELS.keys())


def is_model_loaded(cache_key: str) -> bool:
    """Check if a model is loaded."""
    return cache_key in _LOADED_MODELS


def clear_cache():
    """Clear all cached models."""
    for key in list(_LOADED_MODELS.keys()):
        unload_model(key)
    log.info("All models unloaded")
