"""
Model loader nodes for UniRig - Pre-load and cache ML models for faster inference.

These nodes download and cache models so subsequent inference runs are faster.
"""

import logging
import os
import sys
import yaml
from pathlib import Path

log = logging.getLogger("unirig")

# Lazy import for Box to avoid import errors before install.py runs
def _get_box():
    """Lazy import for python-box."""
    from box import Box
    return Box

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .base import UNIRIG_PATH, UNIRIG_MODELS_DIR
except ImportError:
    from base import UNIRIG_PATH, UNIRIG_MODELS_DIR

# Global model cache (for config dicts)
_MODEL_CACHE = {}

# In-process model cache module
_MODEL_CACHE_MODULE = None

# Store import errors for better error messages
_MODEL_CACHE_IMPORT_ERROR = None

try:
    from . import model_cache as _MODEL_CACHE_MODULE
except ImportError as e:
    _MODEL_CACHE_IMPORT_ERROR = e
    error_msg = str(e).lower()
    if "spconv" in error_msg:
        log.error("spconv is not installed. Install with: pip install spconv-cu121 (match your CUDA version)")
    elif "torch_scatter" in error_msg:
        log.error("torch-scatter is not installed. Install with: pip install torch-scatter")
    elif "torch_cluster" in error_msg:
        log.error("torch-cluster is not installed. Install with: pip install torch-cluster")
    else:
        log.error("Failed to load model cache: %s", e)
    _MODEL_CACHE_MODULE = None
except Exception as e:
    _MODEL_CACHE_IMPORT_ERROR = e
    log.error("Failed to load model cache: %s", e, exc_info=True)
    _MODEL_CACHE_MODULE = None


def _get_model_cache():
    """Get the in-process model cache module."""
    return _MODEL_CACHE_MODULE


def get_model_cache_error():
    """Get the error that caused model cache loading to fail."""
    return _MODEL_CACHE_IMPORT_ERROR


def _load_yaml_config(config_path: str):
    """Load a YAML config file."""
    Box = _get_box()
    if config_path.endswith('.yaml'):
        config_path = config_path.removesuffix('.yaml')
    config_path += '.yaml'
    return Box(yaml.safe_load(open(config_path, 'r')))


class UniRigLoadSkeletonModel:
    """
    Load and cache the UniRig skeleton extraction model.

    This pre-downloads the model weights and prepares configuration
    for faster skeleton inference. Connect this to UniRigExtractSkeleton
    to avoid model reload on each run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "apozz/UniRig-safetensors",
                    "tooltip": "HuggingFace model ID for skeleton model"
                }),
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model cached on GPU for faster inference. Disable to offload to CPU after inference (saves VRAM)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_SKELETON_MODEL",)
    RETURN_NAMES = ("skeleton_model",)
    FUNCTION = "load_model"
    CATEGORY = "UniRig/Models"

    def load_model(self, model_id="apozz/UniRig-safetensors", cache_to_gpu=True):
        """Load and cache skeleton model configuration."""
        log.info("Loading skeleton model: %s", model_id)
        log.info("GPU caching: %s", 'enabled' if cache_to_gpu else 'disabled (will offload to CPU)')

        cache_key = f"skeleton_{model_id}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            cached_model = _MODEL_CACHE[cache_key]
            # Update cache_to_gpu setting in case it changed
            cached_model["cache_to_gpu"] = cache_to_gpu
            log.info("Using cached model configuration")

            # If cache_to_gpu is enabled but model is not in GPU memory, load it now
            if cache_to_gpu and cached_model.get("model_cache_key") is None:
                model_cache = _get_model_cache()
                if model_cache:
                    log.info("Loading model into GPU memory (delayed load)...")
                    try:
                        model_cache_key = model_cache.load_model_into_memory(
                            model_type="skeleton",
                            task_config_path=cached_model.get("task_config_path"),
                            cache_to_gpu=True
                        )
                        cached_model["model_cache_key"] = model_cache_key
                        log.info("Model loaded into GPU memory")
                    except Exception as e:
                        log.warning("Failed to load into GPU memory: %s", e)
                
            return (cached_model,)

                # Pre-download model weights
        try:
            from .unirig.src.inference.download import download

            # Load task config to get checkpoint path
            task_config_path = os.path.join(
                UNIRIG_PATH,
                "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
            )
            task_config = _load_yaml_config(task_config_path)

            # Download checkpoint if needed
            checkpoint_path = task_config.get('resume_from_checkpoint', None)
            if checkpoint_path:
                log.info("Downloading/verifying checkpoint...")
                local_checkpoint = download(checkpoint_path)
                log.info("Checkpoint ready: %s", local_checkpoint)
            else:
                local_checkpoint = None

            # Load model config
            model_config_name = task_config.components.get('model', None)
            if model_config_name:
                model_config = _load_yaml_config(
                    os.path.join(UNIRIG_PATH, 'configs/model', model_config_name)
                )
            else:
                model_config = {}

            # Load tokenizer config
            tokenizer_config_name = task_config.components.get('tokenizer', None)
            if tokenizer_config_name:
                tokenizer_config = _load_yaml_config(
                    os.path.join(UNIRIG_PATH, 'configs/tokenizer', tokenizer_config_name)
                )
            else:
                tokenizer_config = None

            # Create model wrapper
            model_wrapper = {
                "type": "skeleton",
                "model_id": model_id,
                "task_config_path": task_config_path,
                "checkpoint_path": local_checkpoint,
                "model_config": model_config.to_dict() if hasattr(model_config, 'to_dict') else dict(model_config),
                "tokenizer_config": tokenizer_config.to_dict() if tokenizer_config and hasattr(tokenizer_config, 'to_dict') else (dict(tokenizer_config) if tokenizer_config else None),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": True,
                "cache_to_gpu": cache_to_gpu,
                "model_cache_key": None,
            }

            # If cache_to_gpu is enabled, load model into GPU memory
            if cache_to_gpu:
                model_cache = _get_model_cache()
                if model_cache:
                    log.info("Loading model into GPU memory...")
                    try:
                        model_cache_key = model_cache.load_model_into_memory(
                            model_type="skeleton",
                            task_config_path=task_config_path,
                            cache_to_gpu=True
                        )
                        model_wrapper["model_cache_key"] = model_cache_key
                        log.info("Model loaded into GPU memory")
                    except Exception as e:
                        log.warning("Failed to load into GPU memory: %s", e)
                        log.warning("Will use subprocess fallback")
                else:
                    log.warning("Model cache not available")

            # Cache it
            _MODEL_CACHE[cache_key] = model_wrapper

            log.info("Model configuration cached successfully")
            log.info("Checkpoint: %s", local_checkpoint)

            return (model_wrapper,)

        except Exception as e:
            log.error("Error loading model: %s", e, exc_info=True)

            # Return minimal config that will trigger full load in inference
            model_wrapper = {
                "type": "skeleton",
                "model_id": model_id,
                "task_config_path": os.path.join(
                    UNIRIG_PATH,
                    "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
                ),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": False,
                "model_cache_key": None,
            }
            return (model_wrapper,)


class UniRigLoadSkinningModel:
    """
    Load and cache the UniRig skinning weight prediction model.

    This pre-downloads the model weights and prepares configuration
    for faster skinning inference. Connect this to UniRigApplySkinningML
    to avoid model reload on each run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "apozz/UniRig-safetensors",
                    "tooltip": "HuggingFace model ID for skinning model"
                }),
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model cached on GPU for faster inference. Disable to offload to CPU after inference (saves VRAM)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_SKINNING_MODEL",)
    RETURN_NAMES = ("skinning_model",)
    FUNCTION = "load_model"
    CATEGORY = "UniRig/Models"

    def load_model(self, model_id="apozz/UniRig-safetensors", cache_to_gpu=True):
        """Load and cache skinning model configuration."""
        log.info("Loading skinning model: %s", model_id)
        log.info("GPU caching: %s", 'enabled' if cache_to_gpu else 'disabled (will offload to CPU)')

        cache_key = f"skinning_{model_id}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            cached_model = _MODEL_CACHE[cache_key]
            # Update cache_to_gpu setting in case it changed
            cached_model["cache_to_gpu"] = cache_to_gpu
            log.info("Using cached model configuration")

            # If cache_to_gpu is enabled but model is not in GPU memory, load it now
            if cache_to_gpu and cached_model.get("model_cache_key") is None:
                model_cache = _get_model_cache()
                if model_cache:
                    log.info("Loading model into GPU memory (delayed load)...")
                    try:
                        model_cache_key = model_cache.load_model_into_memory(
                            model_type="skinning",
                            task_config_path=cached_model.get("task_config_path"),
                            cache_to_gpu=True
                        )
                        cached_model["model_cache_key"] = model_cache_key
                        log.info("Model loaded into GPU memory")
                    except Exception as e:
                        log.warning("Failed to load into GPU memory: %s", e)
                        
            return (cached_model,)

                # Pre-download model weights
        try:
            from .unirig.src.inference.download import download

            # Load task config to get checkpoint path
            task_config_path = os.path.join(
                UNIRIG_PATH,
                "configs/task/quick_inference_unirig_skin.yaml"
            )
            task_config = _load_yaml_config(task_config_path)

            # Download checkpoint if needed
            checkpoint_path = task_config.get('resume_from_checkpoint', None)
            if checkpoint_path:
                log.info("Downloading/verifying checkpoint...")
                local_checkpoint = download(checkpoint_path)
                log.info("Checkpoint ready: %s", local_checkpoint)
            else:
                local_checkpoint = None

            # Load model config
            model_config_name = task_config.components.get('model', None)
            if model_config_name:
                model_config = _load_yaml_config(
                    os.path.join(UNIRIG_PATH, 'configs/model', model_config_name)
                )
            else:
                model_config = {}

            # Create model wrapper
            model_wrapper = {
                "type": "skinning",
                "model_id": model_id,
                "task_config_path": task_config_path,
                "checkpoint_path": local_checkpoint,
                "model_config": model_config.to_dict() if hasattr(model_config, 'to_dict') else dict(model_config),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": True,
                "cache_to_gpu": cache_to_gpu,
                "model_cache_key": None,
            }

            # If cache_to_gpu is enabled, load model into GPU memory
            if cache_to_gpu:
                model_cache = _get_model_cache()
                if model_cache:
                    log.info("Loading model into GPU memory...")
                    try:
                        model_cache_key = model_cache.load_model_into_memory(
                            model_type="skinning",
                            task_config_path=task_config_path,
                            cache_to_gpu=True
                        )
                        model_wrapper["model_cache_key"] = model_cache_key
                        log.info("Model loaded into GPU memory")
                    except Exception as e:
                        log.warning("Failed to load into GPU memory: %s", e)
                        log.warning("Will use subprocess fallback")
                else:
                    log.warning("Model cache not available")

            # Cache it
            _MODEL_CACHE[cache_key] = model_wrapper

            log.info("Model configuration cached successfully")
            log.info("Checkpoint: %s", local_checkpoint)

            return (model_wrapper,)

        except Exception as e:
            log.error("Error loading model: %s", e, exc_info=True)

            # Return minimal config
            model_wrapper = {
                "type": "skinning",
                "model_id": model_id,
                "task_config_path": os.path.join(
                    UNIRIG_PATH,
                    "configs/task/quick_inference_unirig_skin.yaml"
                ),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": False,
                "model_cache_key": None,
            }
            return (model_wrapper,)


class UniRigLoadModel:
    """
    Load and cache both UniRig models (skeleton + skinning) for the rigging pipeline.

    This node downloads and caches both model weights so subsequent inference runs
    are faster. When cache_to_gpu is enabled, models are loaded directly into GPU memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models cached on GPU for faster inference. Disable to save VRAM (models will be loaded on-demand)."
                }),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_models"
    CATEGORY = "UniRig"

    def load_models(self, cache_to_gpu=True, precision="auto"):
        """Load and cache both skeleton and skinning models."""
        log.info("Loading UniRig models...")
        log.info("GPU caching: %s", 'enabled' if cache_to_gpu else 'disabled')

        model_id = "apozz/UniRig-safetensors"

        # Load skeleton model
        skeleton_loader = UniRigLoadSkeletonModel()
        skeleton_result = skeleton_loader.load_model(model_id=model_id, cache_to_gpu=cache_to_gpu)
        skeleton_model = skeleton_result[0]

        # Load skinning model
        skinning_loader = UniRigLoadSkinningModel()
        skinning_result = skinning_loader.load_model(model_id=model_id, cache_to_gpu=cache_to_gpu)
        skinning_model = skinning_result[0]

        # Combine into single model dict
        combined_model = {
            "skeleton_model": skeleton_model,
            "skinning_model": skinning_model,
            "model_id": model_id,
            "cache_to_gpu": cache_to_gpu,
            "precision": precision,
        }

        log.info("Both models loaded successfully")
        return (combined_model,)


def clear_model_cache():
    """Clear the global model cache."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    log.info("Model cache cleared")


def get_cached_models():
    """Get list of cached model keys."""
    return list(_MODEL_CACHE.keys())
