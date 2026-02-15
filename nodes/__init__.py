"""UniRig Nodes."""

# Patch torch.nn.init to skip wasteful random weight initialization.
# All models load checkpoint weights via load_state_dict() immediately after
# construction, so the default kaiming/xavier/etc. init is pure overhead.
# Safe: this file only runs inside the comfy-env isolated subprocess.
import torch.nn.init as _init

def _noop(tensor, *args, **kwargs):
    return tensor

for _fn in (
    "kaiming_uniform_", "kaiming_normal_",
    "xavier_uniform_", "xavier_normal_",
    "uniform_", "normal_", "trunc_normal_",
    "ones_", "zeros_", "constant_",
    "orthogonal_",
):
    if hasattr(_init, _fn):
        setattr(_init, _fn, _noop)

from .mesh_io import UniRigLoadMesh, UniRigSaveMesh
from .model_loaders import UniRigLoadModel
from .auto_rig import UniRigAutoRig
from .skeleton_extraction import UniRigExtractSkeletonNew
from .skinning import UniRigApplySkinningMLNew
from .mia_model_loader import MIALoadModel
from .mia_auto_rig import MIAAutoRig
from .animation import UniRigApplyAnimation
from .skeleton_io import (
    UniRigLoadRiggedMesh,
    UniRigPreviewRiggedMesh,
    UniRigExportPosedFBX,
    UniRigViewRigging,
    UniRigDebugSkeleton,
    UniRigCompareSkeletons,
)
from .rest_pose_node import UniRigExtractRestPose
from .orientation_check import UniRigOrientationCheck

NODE_CLASS_MAPPINGS = {
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
    "UniRigLoadModel": UniRigLoadModel,
    "UniRigAutoRig": UniRigAutoRig,
    "UniRigExtractSkeletonNew": UniRigExtractSkeletonNew,
    "UniRigApplySkinningMLNew": UniRigApplySkinningMLNew,
    "MIALoadModel": MIALoadModel,
    "MIAAutoRig": MIAAutoRig,
    "UniRigApplyAnimation": UniRigApplyAnimation,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigExportPosedFBX": UniRigExportPosedFBX,
    "UniRigViewRigging": UniRigViewRigging,
    "UniRigDebugSkeleton": UniRigDebugSkeleton,
    "UniRigCompareSkeletons": UniRigCompareSkeletons,
    "UniRigExtractRestPose": UniRigExtractRestPose,
    "UniRigOrientationCheck": UniRigOrientationCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
    "UniRigLoadModel": "UniRig: Load Model",
    "UniRigAutoRig": "UniRig: Auto Rig",
    "UniRigExtractSkeletonNew": "UniRig: Extract Skeleton",
    "UniRigApplySkinningMLNew": "UniRig: Apply Skinning ML",
    "MIALoadModel": "MIA: Load Model",
    "MIAAutoRig": "MIA: Auto Rig",
    "UniRigApplyAnimation": "UniRig: Apply Animation",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigExportPosedFBX": "UniRig: Export Posed FBX",
    "UniRigViewRigging": "UniRig: View Rigging",
    "UniRigDebugSkeleton": "UniRig: Debug Skeleton",
    "UniRigCompareSkeletons": "UniRig: Compare Skeletons",
    "UniRigExtractRestPose": "UniRig: Extract Rest Pose",
    "UniRigOrientationCheck": "UniRig: Orientation Check",
}
