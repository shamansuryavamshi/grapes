"""Model definition and checkpoint handling."""

from iwnet.model.architecture import (
    ChannelAttention,
    FeatureShapeError,
    IWAttention,
    IWNET,
    SpatialAttention,
    build_model,
    describe_model,
)
from iwnet.model.checkpoint import (
    CheckpointError,
    build_metadata,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "IWAttention",
    "FeatureShapeError",
    "IWNET",
    "build_model",
    "describe_model",
    "CheckpointError",
    "save_checkpoint",
    "load_checkpoint",
    "inspect_checkpoint",
    "build_metadata",
]
