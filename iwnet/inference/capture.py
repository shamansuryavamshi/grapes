"""Temporary, non-invasive capture of IWNET intermediates for visualization.

Why this exists
---------------
The visualization feature must *observe* the production pipeline, never fork it.
The cheapest faithful way to do that is to hang read-only forward hooks on the
modules that already compute the quantities we want to show, let the existing
``IWNET.forward`` run exactly once, and read what they produced.

What it does not do
-------------------
This module never:

* constructs, replaces, re-initialises or deep-copies the model,
* touches a parameter, buffer or ``training`` flag,
* runs an extra forward pass,
* or alters a single value the network computes.

A hook whose body only stores a reference to the module's own output is a pure
observer. It therefore follows that ``predict_pil`` and ``predict_pil_traced``
return **identical logits and identical probabilities**, from the same single
forward pass. ``tests/test_visualization.py`` asserts exactly that.

Every hook is removed in a ``finally`` block, even if the forward pass raises, so
a failed visualization can never leave the model permanently instrumented.

Which tensors are captured, and why these
-----------------------------------------
IWNET applies, per stage ``i``, ``p_i = proj_i(att_i(f_i))`` where
``att_i = IWAttention = ChannelAttention -> SpatialAttention -> IW gate``. The
modules below expose the *pre-activation* value of each quantity, so applying the
module's own final nonlinearity (sigmoid / softmax) afterwards reproduces the
exact tensor the model used, rather than re-deriving it from inputs:

===============================  =========================================
captured                         exact model quantity reproduced
===============================  =========================================
``att{i}.ca.fc`` last Linear     ``sigmoid(...)`` = ChannelAttention weights
``att{i}.sa.conv``               ``sigmoid(...)`` = SpatialAttention map
``att{i}.iw``   last Linear      ``sigmoid(...)`` = IW per-sample gate
``proj{i}``                      1x1-projected feature map for scale i
``scale_w``                      ``softmax(...)`` = adaptive scale weights
===============================  =========================================
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn

__all__ = [
    "STAGE_COUNT",
    "IntermediateTrace",
    "capture_intermediates",
    "last_linear",
]

#: IWNET fuses exactly three backbone stages. Kept as a constant rather than
#: hardcoded so a future architecture change surfaces here instead of silently
#: producing a partial trace.
STAGE_COUNT = 3


def last_linear(module: nn.Module) -> nn.Linear:
    """Return the final :class:`torch.nn.Linear` inside ``module``.

    Resolved by type rather than by index, so it keeps working if a block gains
    or loses a layer. Raises if there is no ``Linear`` at all.
    """
    for sub in reversed(list(module.modules())):
        if isinstance(sub, nn.Linear):
            return sub
    raise AttributeError(f"No nn.Linear found inside {type(module).__name__}.")


def _stash(store: dict[str, Any], key: str, output: Any) -> None:
    """Hook body: keep one batch element, detach, and drop the autograd graph.

    Batch size is 1 for single-image inference; slicing defensively keeps this
    safe if that ever changes.
    """
    tensor = output[0] if isinstance(output, tuple) else output
    if isinstance(tensor, torch.Tensor):
        store[key] = tensor.detach()[:1].clone()
    else:
        store[key] = tensor


@contextmanager
def capture_intermediates(model: nn.Module) -> Iterator[dict[str, Any]]:
    """Yield a dict that is populated while ``model`` runs its next forward pass.

    The dict is filled *during* the ``with`` body; read it after the forward pass
    has completed but still inside the ``with`` block, or copy what you need
    first. Hooks are always detached on exit, including on exception.
    """
    if not hasattr(model, "scale_w"):
        raise AttributeError(
            "capture_intermediates expects an IWNET instance; got "
            f"{type(model).__name__}."
        )

    store: dict[str, Any] = {}
    handles: list[Any] = []

    def hook(key: str):
        def _fn(_module, _inputs, output):
            _stash(store, key, output)

        return _fn

    try:
        for stage in range(1, STAGE_COUNT + 1):
            att = getattr(model, f"att{stage}")
            handles.append(
                last_linear(att.ca.fc).register_forward_hook(hook(f"ca{stage}"))
            )
            handles.append(att.sa.conv.register_forward_hook(hook(f"sa{stage}")))
            handles.append(
                last_linear(att.iw).register_forward_hook(hook(f"iw{stage}"))
            )
            proj = getattr(model, f"proj{stage}")
            handles.append(proj.register_forward_hook(hook(f"proj{stage}")))
        handles.append(model.scale_w.register_forward_hook(hook("scale_w")))

        yield store
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()


def _as_numpy(value: Any):
    import numpy as np

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


class IntermediateTrace:
    """The captured tensors, converted to the exact model quantities.

    Construct from the dict yielded by :func:`capture_intermediates` after the
    forward pass has run. Every array is a *copy*, so the trace stays valid after
    the source tensors have been freed.

    This class holds NumPy arrays only - no autograd graph, no reference to the
    model, and no reference to the intermediate feature tensors.
    """

    __slots__ = (
        "feature_channels",
        "embedding_dim",
        "out_indices",
        "num_classes",
        "scale_weights",
        "channel_attention",
        "spatial_attention",
        "iw_gates",
        "projected",
        "projected_shapes",
    )

    def __init__(self, store: dict[str, Any], model: nn.Module) -> None:
        import numpy as np

        self.feature_channels = tuple(int(c) for c in model.feature_channels)
        self.embedding_dim = int(model.D)
        self.out_indices = tuple(int(i) for i in model.out_indices)
        self.num_classes = int(model.num_classes)

        raw_scale = store.get("scale_w")
        # `forward` computes exactly: softmax(self.scale_w(global_features), dim=1)
        self.scale_weights = (
            np.exp(_as_numpy(raw_scale) - np.max(_as_numpy(raw_scale)))
            / np.exp(_as_numpy(raw_scale) - np.max(_as_numpy(raw_scale))).sum(axis=1, keepdims=True)
            if raw_scale is not None
            else None
        )

        self.channel_attention: list[Any] = []
        self.spatial_attention: list[Any] = []
        self.iw_gates: list[Any] = []
        self.projected: list[Any] = []
        self.projected_shapes: list[tuple[int, int, int, int]] = []

        for stage in range(1, STAGE_COUNT + 1):
            # ChannelAttention.forward: torch.sigmoid(self.fc(x.mean(dim=[2, 3])))
            ca = store.get(f"ca{stage}")
            self.channel_attention.append(
                _sigmoid(_as_numpy(ca)) if ca is not None else None
            )

            # SpatialAttention.forward: torch.sigmoid(self.conv(pooled))
            sa = store.get(f"sa{stage}")
            self.spatial_attention.append(
                _sigmoid(_as_numpy(sa)) if sa is not None else None
            )

            # IWAttention gate: torch.sigmoid(linear(...))
            iw = store.get(f"iw{stage}")
            self.iw_gates.append(
                _sigmoid(_as_numpy(iw)) if iw is not None else None
            )

            proj = store.get(f"proj{stage}")
            if proj is not None:
                array = _as_numpy(proj)
                self.projected.append(array[0])  # (C, H, W)
                self.projected_shapes.append(tuple(int(v) for v in array.shape))
            else:
                self.projected.append(None)
                self.projected_shapes.append(None)

    @property
    def is_complete(self) -> bool:
        """True when every stage produced a channel, spatial and feature tensor."""
        return all(
            item is not None
            for group in (self.channel_attention, self.spatial_attention, self.projected)
            for item in group
        )

    def release(self) -> None:
        """Drop every array. Called once the visualization payload is built."""
        self.channel_attention = [None] * STAGE_COUNT
        self.spatial_attention = [None] * STAGE_COUNT
        self.projected = [None] * STAGE_COUNT
        self.projected_shapes = [None] * STAGE_COUNT
        self.scale_weights = None


def _sigmoid(array):
    """Numerically stable logistic, matching ``torch.sigmoid``."""
    import numpy as np

    out = np.empty_like(array, dtype=np.float32)
    positive = array >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_array = np.exp(array[~positive])
    out[~positive] = exp_array / (1.0 + exp_array)
    return out
