"""Compact, display-only representations of the production inference pipeline.

Contract with the rest of the project
-------------------------------------
* **One pipeline.** Every image here is derived from the objects the production
  predictor already used: the same PIL image, the same
  ``build_eval_transform`` Compose instance, and the same tensors the model
  produced during the same single forward pass. There is no second resize, no
  second normalization and no second model anywhere in this file.
* **Predictions are never touched.** Nothing here can alter logits,
  probabilities, the predicted class or the reported timings.
* **Visualization only.** The inverse-normalized and clipped renders exist purely
  so a human can look at a tensor. They are PNGs returned to the browser and are
  never fed back to the model.

Wording discipline
------------------
Per the project's explanation rules, the captions describe what a tensor *is*,
not what the model "saw". Spatial attention is described as relative spatial
weighting; nothing here claims the model detected a specific lesion.

Size discipline
---------------
Raw feature tensors are never returned. Projected maps are reduced to
per-channel 2D planes, attention maps are already 1- or 2-dimensional by
construction, and every image is emitted as a PNG data URL. Typical payload is a
few hundred kilobytes.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Sequence

import numpy as np

__all__ = ["build_visualization", "encode_png_data_url"]

#: Longest edge used for any *display* render. The prediction itself is computed
#: at full resolution; this only bounds how big the returned PNG is.
DISPLAY_MAX_EDGE = 448

#: A short perceptual ramp for heatmaps (sampled from matplotlib's ``inferno``).
#: Reimplemented with numpy so the API does not pay matplotlib's import cost.
_INFERNO_ANCHORS = np.array(
    [
        [0, 0, 4],
        [22, 11, 57],
        [59, 15, 112],
        [100, 26, 128],
        [140, 41, 129],
        [183, 55, 121],
        [222, 73, 104],
        [244, 109, 67],
        [252, 165, 10],
        [252, 255, 164],
    ],
    dtype=np.float64,
)
_INFERNO_X = np.linspace(0.0, 1.0, len(_INFERNO_ANCHORS))

#: Window the normalized tensor is clipped to when rendered. ImageNet-normalized
#: tensors mostly live in roughly [-2.5, 2.5]; clipping shows structure without
#: letting a handful of saturated pixels flatten the rest of the range.
NORMALIZE_CLIP = 2.5


# ── encoding helpers ────────────────────────────────────────────────────────
def _ramp(values: np.ndarray) -> np.ndarray:
    """Map scalars in [0, 1] to RGB uint8 along the perceptual ramp."""
    flat = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).ravel()
    rgb = np.stack([np.interp(flat, _INFERNO_X, _INFERNO_ANCHORS[:, c]) for c in range(3)], axis=1)
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(np.asarray(values).shape + (3,))


def _fit(image, max_edge: int = DISPLAY_MAX_EDGE):
    """Return ``(image, downscaled)`` without ever enlarging the source."""
    from PIL import Image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image, False
    scale = max_edge / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.BILINEAR), True


def encode_png_data_url(image) -> str:
    """Encode a PIL image as an inline ``data:image/png;base64,...`` URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _array_to_image(image_or_array):
    """Build a PIL image from either a float array in [0, 1] or a uint8 array.

    Integer input is treated as already 0-255. Handling both explicitly removes
    a silent-corruption trap: scaling a uint8 array by 255 overflows and yields
    plausible-looking noise rather than an error.
    """
    from PIL import Image

    if hasattr(image_or_array, "convert") and hasattr(image_or_array, "size"):
        return image_or_array

    data = np.asarray(image_or_array)
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32)
        if float(data.max()) > 1.0:
            data = data / 255.0
    if data.ndim == 2:
        data = data[..., None]
    if data.shape[0] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
        data = np.transpose(data, (1, 2, 0))
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    if data.shape[-1] == 4:
        data = data[..., :3]
    if data.shape[-1] != 3:
        # Any other channel count (e.g. a montage tile) -> mean over channels.
        data = np.repeat(data.mean(axis=-1, keepdims=True), 3, axis=-1)
    data = np.clip(data, 0.0, 1.0)
    return Image.fromarray((data * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _resample(array: np.ndarray, size: int) -> np.ndarray:
    """Reduce to a 2D plane and resize it to ``size`` x ``size`` for display only.

    Accepts any ndim: leading batch/channel axes are squeezed until 2D.
    """
    from PIL import Image

    plane = np.asarray(array, dtype=np.float32)
    while plane.ndim > 2:
        plane = plane[0]
    lo, hi = float(np.min(plane)), float(np.max(plane))
    scaled = (plane - lo) / (hi - lo) if hi > lo else np.zeros_like(plane)
    image = Image.fromarray((scaled * 255.0 + 0.5).astype(np.uint8), mode="L")
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def _normalize_plane(plane: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(plane)), float(np.max(plane))
    return (plane - lo) / (hi - lo) if hi > lo else np.zeros_like(plane, dtype=np.float32)


#: Pixels of background between montage tiles, so panels stay visually distinct.
MONTAGE_GAP = 3


def _montage(tiles: Sequence[np.ndarray], columns: int) -> np.ndarray:
    """Tile 2D planes into one ``(rows*cell, columns*cell, 3)`` float array in [0, 1].

    Returns floats rather than uint8 so every display array in this module shares
    one convention and no caller can accidentally rescale an already-scaled image.
    """
    tiles = [np.asarray(t, dtype=np.float32) for t in tiles]
    if not tiles:
        return np.zeros((MONTAGE_GAP, MONTAGE_GAP, 3), dtype=np.float32)

    side = max(int(t.shape[0]) for t in tiles)
    columns = max(1, min(columns, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    cell = side + MONTAGE_GAP
    out = np.zeros((rows * cell, columns * cell, 3), dtype=np.float32)

    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        placed = _resample(tile, side)
        rgb = _ramp(placed) if placed.ndim == 2 else np.asarray(placed)[..., :3]
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        rgb = np.clip(rgb[..., :3], 0.0, 255.0) / 255.0
        y, x = r * cell, c * cell
        out[y : y + side, x : x + side, :] = rgb
    return out


# ── stage builders ──────────────────────────────────────────────────────────
def _stage(title: str, caption: str, available: bool = True, **extra: Any) -> dict:
    return {"title": title, "caption": caption, "available": available, **extra}


def _resized_pil(image, transform):
    """Re-apply only the geometry ops of the production transform.

    ``build_eval_transform`` is ``Resize -> CenterCrop -> ToTensor -> Normalize``.
    We replay the *same* ``Resize`` and ``CenterCrop`` objects held by the
    *same* Compose instance the predictor uses, which yields exactly the image
    those ops produced, before tensor conversion. If the pipeline ever stops
    exposing those ops, the stage reports unavailable rather than guessing.
    """
    ops = getattr(getattr(transform, "transforms", None), "__iter__", None)
    if ops is None:
        return None
    geometry = [op for op in transform.transforms if type(op).__name__ in {"Resize", "CenterCrop"}]
    if len(geometry) != 2:
        return None
    try:
        working = image
        for op in geometry:
            working = op(working)
        return working
    except Exception:  # pragma: no cover - defensive
        return None


def build_visualization(
    *,
    image,
    transform,
    normalized_tensor,
    trace,
    prediction,
    include_original: bool = True,
) -> dict[str, Any]:
    """Assemble the visualization payload.

    Parameters
    ----------
    image:
        The decoded RGB PIL image handed to the predictor.
    transform:
        The predictor's own ``build_eval_transform`` Compose.
    normalized_tensor:
        The ``(1, 3, H, W)`` normalized tensor the model actually received.
    trace:
        An :class:`~iwnet.inference.capture.IntermediateTrace` from the *same*
        forward pass that produced ``prediction``.
    prediction:
        The :class:`~iwnet.inference.predictor.Prediction` for that pass.
    include_original:
        Set False to omit the submitted image (used by ``image_path`` callers,
        where the browser already has that thumbnail).
    """
    from iwnet.data.dataset import imagenet_norm

    notes: list[str] = []
    stages: dict[str, Any] = {}
    image_size = int(getattr(image, "size", (0, 0))[0] or 224)

    # ── 1. original ─────────────────────────────────────────────────────────
    if include_original:
        fitted, downscaled = _fit(image)
        stages["original"] = _stage(
            "Original Image",
            "The exact image submitted, before any resizing. Display is capped at "
            f"{DISPLAY_MAX_EDGE}px for transfer speed; the model saw the file as sent.",
            image=encode_png_data_url(fitted),
            width=image.size[0],
            height=image.size[1],
            display_downscaled=downscaled,
        )
    else:
        stages["original"] = _stage(
            "Original Image",
            "Already displayed by the caller (this prediction was requested by "
            "server-side dataset path rather than an upload).",
            image=None,
            available=False,
        )

    # ── 2. resized ──────────────────────────────────────────────────────────
    resized = _resized_pil(image, transform)
    if resized is not None:
        stages["resized"] = _stage(
            f"Resized to {resized.size[0]} × {resized.size[1]}",
            "Produced by the first two steps of the production transform "
            "(Resize then CenterCrop), replayed with that exact transform object.",
            image=encode_png_data_url(resized),
            width=resized.size[0],
            height=resized.size[1],
            display_downscaled=False,
        )
    else:
        stages["resized"] = _stage(
            "Resized",
            "The production transform no longer exposes a plain Resize/CenterCrop "
            "pair, so the intermediate image was not rendered rather than approximated.",
            image=None,
            available=False,
        )
        notes.append("Resized-image stage unavailable: transform geometry not exposed.")

    # ── 3. normalized ───────────────────────────────────────────────────────
    array = np.asarray(normalized_tensor, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 3:
        clipped = np.clip(array, -NORMALIZE_CLIP, NORMALIZE_CLIP)
        clipped = (clipped + NORMALIZE_CLIP) / (2.0 * NORMALIZE_CLIP)
        mean, std = imagenet_norm()
        restored = np.clip(array * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32), 0.0, 1.0)
        stages["normalized"] = _stage(
            "Visualization of Normalized Input",
            f"The literal normalized tensor, clipped to "
            f"[-{NORMALIZE_CLIP}, {NORMALIZE_CLIP}]. This is the array the model "
            "receives, not the image the eye expects.",
            image=encode_png_data_url(_array_to_image(np.transpose(clipped, (2, 0, 1)))),
            inverse_image=encode_png_data_url(_array_to_image(np.transpose(restored, (2, 0, 1)))),
            inverse_caption=(
                "Display-only inverse of the normalization (x·std + mean). It recovers "
                "the resized RGB crop and exists solely for human comparison. It is "
                "never fed back into the model."
            ),
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            display_downscaled=False,
        )
    else:
        stages["normalized"] = _stage(
            "Visualization of Normalized Input",
            "The normalized tensor had an unexpected layout and was not rendered.",
            image=None,
            available=False,
        )
        notes.append("Normalized-input stage unavailable: unexpected tensor layout.")

    # ── 4. feature representation ───────────────────────────────────────────
    if trace is not None and trace.is_complete:
        per_stage = []
        tiles: list[np.ndarray] = []
        for index, projected in enumerate(trace.projected):
            if projected is None:
                per_stage.append(None)
                continue
            magnitude = np.abs(projected).mean(axis=(1, 2))
            order = np.argsort(magnitude)[::-1][:8]
            planes = [_normalize_plane(projected[int(c)]) for c in order]
            tiles.extend(planes)
            per_stage.append(
                {
                    "stage": index + 1,
                    "channels": int(projected.shape[0]),
                    "height": int(projected.shape[1]),
                    "width": int(projected.shape[2]),
                    "top_channels": [int(c) for c in order],
                    "top_channel_mean_abs": [round(float(magnitude[int(c)]), 6) for c in order],
                }
            )
        montage = _montage(tiles, 8)
        stages["features"] = _stage(
            "Intermediate Feature Representation",
            "The eight highest-mean-activation channels of each projected multi-scale "
            "feature map, after the 1×1 projection to the embedding dimension. Each "
            "channel is rescaled independently for display and tiled together. This is "
            "an activation summary, not a saliency map.",
            image=encode_png_data_url(_array_to_image(np.transpose(montage, (2, 0, 1)))),
            stages=per_stage,
            embedding_dim=trace.embedding_dim,
            montage_columns=8,
            display_downscaled=False,
        )
    else:
        stages["features"] = _stage(
            "Intermediate Feature Representation",
            "Projected feature maps were not captured for this pass.",
            image=None,
            available=False,
        )
        notes.append("Feature-representation stage unavailable: no captured feature maps.")

    # ── 5. channel attention ────────────────────────────────────────────────
    if trace is not None and all(item is not None for item in trace.channel_attention):
        per_stage = []
        for index, weights in enumerate(trace.channel_attention):
            flat = np.asarray(weights, dtype=np.float32).reshape(-1)
            order = np.argsort(flat)[::-1][:12]
            per_stage.append(
                {
                    "stage": index + 1,
                    "channels": int(flat.size),
                    "weights": [round(float(v), 6) for v in flat],
                    "mean": round(float(flat.mean()), 6),
                    "min": round(float(flat.min()), 6),
                    "max": round(float(flat.max()), 6),
                    "top_channels": [int(c) for c in order],
                    "top_weights": [round(float(flat[int(c)]), 6) for c in order],
                }
            )
        stages["channel_attention"] = _stage(
            "Channel Attention",
            "Relative weighting assigned to feature channels by the squeeze-and-excite "
            "module in each stage. Values are sigmoid outputs in (0, 1); 1.0 means a "
            "channel was passed through unattenuated, not that it mattered to the class.",
            stages=per_stage,
            bars_stage=len(per_stage),
        )
    else:
        stages["channel_attention"] = _stage(
            "Channel Attention",
            "Channel-attention weights were not captured for this pass.",
            available=False,
        )
        notes.append("Channel-attention stage unavailable: no captured weights.")

    # ── 6. spatial attention ────────────────────────────────────────────────
    if trace is not None and all(item is not None for item in trace.spatial_attention):
        per_stage = []
        heat_tiles: list[np.ndarray] = []
        reduced_planes: list[np.ndarray] = []
        raw_maps: list[Any] = []
        for index, plane in enumerate(trace.spatial_attention):
            plane = np.asarray(plane, dtype=np.float32)
            while plane.ndim > 2:
                plane = plane[0]
            if plane.size > 32 * 32:
                step = int(np.ceil(plane.shape[0] / 32.0))
                thinned = plane[::step, ::step]
            else:
                thinned = plane
            raw_maps.append(
                {
                    "stage": index + 1,
                    "height": int(plane.shape[0]),
                    "width": int(plane.shape[1]),
                    "values": [round(float(v), 4) for v in thinned.ravel()],
                }
            )
            per_stage.append(
                {
                    "stage": index + 1,
                    "min": round(float(plane.min()), 6),
                    "max": round(float(plane.max()), 6),
                    "mean": round(float(plane.mean()), 6),
                }
            )
            reduced_planes.append(plane)
            heat_tiles.append(_normalize_plane(plane))

        montage = _montage(heat_tiles, len(heat_tiles))
        overlay_url = None
        if resized is not None:
            deepest = _resample(reduced_planes[-1], resized.size[0])
            heat = _ramp(deepest).astype(np.float32)
            base = np.asarray(resized.convert("RGB"), dtype=np.float32)
            if heat.shape[:2] != base.shape[:2]:
                heat = np.asarray(
                    _array_to_image(heat.transpose(2, 0, 1)).resize(
                        (base.shape[1], base.shape[0])
                    ),
                    dtype=np.float32,
                )
            blended = np.clip(base * 0.45 + heat * 0.55, 0, 255).astype(np.uint8)
            from PIL import Image

            overlay_url = encode_png_data_url(Image.fromarray(blended, mode="RGB"))

        stages["spatial_attention"] = _stage(
            "Spatial Attention",
            "Attention visualization — indicates regions receiving stronger spatial "
            "weighting. Brighter means a larger sigmoid value in the spatial-attention "
            "convolution, i.e. the map was multiplied by a higher weight there. It is "
            "not a claim that a specific lesion was detected at a specific pixel.",
            image=overlay_url,
            overlay_caption=(
                f"Stage {len(per_stage)} spatial-attention map composited over the "
                "resized input at 55% opacity."
            ),
            montage=encode_png_data_url(_array_to_image(montage.transpose(2, 0, 1))),
            montage_caption="All three stages side by side, each rescaled for display.",
            stages=per_stage,
            maps=raw_maps,
        )
    else:
        stages["spatial_attention"] = _stage(
            "Spatial Attention",
            "Spatial-attention maps were not captured for this pass.",
            image=None,
            available=False,
        )
        notes.append("Spatial-attention stage unavailable: no captured maps.")

    # ── 7. multi-scale fusion ───────────────────────────────────────────────
    if trace is not None and trace.scale_weights is not None and trace.is_complete:
        weights = np.asarray(trace.scale_weights, dtype=np.float32).reshape(-1)
        scales = []
        tiles = []
        for index, projected in enumerate(trace.projected):
            weight = float(weights[index]) if index < weights.size else 0.0
            scales.append(
                {
                    "stage": index + 1,
                    "backbone_out_index": int(trace.out_indices[index])
                    if index < len(trace.out_indices)
                    else None,
                    "weight": round(weight, 6),
                    "channels": int(projected.shape[0]) if projected is not None else None,
                    "height": int(projected.shape[1]) if projected is not None else None,
                    "width": int(projected.shape[2]) if projected is not None else None,
                }
            )
            if projected is not None:
                tiles.append(projected.mean(axis=0))
        montage = _montage(tiles, len(tiles))
        stages["multi_scale"] = _stage(
            "Multi-Scale Fusion",
            "The actual softmax scale weights IWNET assigned to each backbone stage "
            "for this image, plus the channel-mean of each projected feature map. The "
            "weights are read straight from the model; nothing is invented or renamed.",
            image=encode_png_data_url(_array_to_image(montage.transpose(2, 0, 1))),
            scales=scales,
            weight_sum=round(float(weights.sum()), 6),
            iw_gates=[
                round(float(np.asarray(gate, dtype=np.float32).reshape(-1)[0]), 6)
                for gate in trace.iw_gates
                if gate is not None
            ],
            iw_gate_caption=(
                "IWAttention's per-sample gate. Each collapses the whole feature map "
                "to one scalar, so it scales a sample uniformly and carries no spatial "
                "or channel information."
            ),
        )
    else:
        stages["multi_scale"] = _stage(
            "Multi-Scale Fusion",
            "Adaptive scale weights were not captured for this pass.",
            image=None,
            available=False,
        )
        notes.append("Multi-scale stage unavailable: no captured scale weights.")

    # ── 8. class probabilities ──────────────────────────────────────────────
    ranked = sorted(prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    stages["probabilities"] = _stage(
        "Class Probability Distribution",
        "The seven softmax outputs from this exact forward pass, unchanged. "
        "The top row is the predicted class.",
        # Deliberately NOT rounded. These are the same numbers the top-level
        # `probabilities` field carries; rounding here would create a second,
        # slightly different set of values for the same prediction.
        rows=[
            {
                "name": name,
                "probability": float(value),
                "predicted": name == prediction.predicted_class,
            }
            for name, value in ranked
        ],
        top=prediction.predicted_class,
        matches_response=True,
    )

    missing = sorted(key for key, value in stages.items() if not value.get("available"))
    return {
        "available": True,
        "engine": "activation-capture",
        "source": "single forward pass of the production model",
        "image_size": image_size,
        "stages": stages,
        "unavailable_stages": missing,
        "notes": notes,
        "disclaimer": (
            "Interpretability aid. Activation and attention tensors show how the "
            "network weighted its own features; they are not a causal explanation of "
            "the prediction and should not be read as a diagnosis."
        ),
    }
