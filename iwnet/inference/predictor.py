"""Single-image inference around a cached, once-loaded model.

Design points that matter:

* **The model is built and loaded exactly once** per process and reused. The
  original script reconstructed the whole network on every ``--predict`` call.
* ``pretrained=False`` when constructing the architecture. Every parameter is
  immediately overwritten by the checkpoint's ``state_dict``, so downloading
  ImageNet weights was pure waste - and it made offline inference impossible.
* **Preprocessing is read from the checkpoint** (``image_size``) and built by the
  same function the training evaluation path uses, so the two cannot drift.
* Everything runs under ``torch.inference_mode()`` with ``model.eval()``.
* Softmax output is reported as *model confidence*, not a calibrated
  probability. No calibration was performed, so it is not one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from iwnet.classes import describe_class, is_rejection_class
from iwnet.config import FINAL_CLASSES, Config
from iwnet.model.checkpoint import CheckpointError, load_checkpoint
from iwnet.utils import get_device, get_logger

__all__ = [
    "Prediction",
    "InferenceError",
    "Predictor",
    "get_predictor",
    "reset_predictor",
    "predict_image_path",
]

log = get_logger("inference.predictor")


class InferenceError(RuntimeError):
    """Raised when inference cannot be performed. Never leaks internals upward."""


@dataclass
class Prediction:
    """One prediction, in a form both the CLI and the API can use directly."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    image_size: int
    inference_ms: float
    preprocessing_ms: float = 0.0
    is_rejection: bool = False
    class_info: dict[str, str] = field(default_factory=dict)
    model_trained_at: str | None = None
    dataset_fingerprint: str | None = None

    def to_dict(self) -> dict:
        return {
            "prediction": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "is_rejection": self.is_rejection,
            "class_info": self.class_info,
            "model": {
                "image_size": self.image_size,
                "trained_at_utc": self.model_trained_at,
                "dataset_fingerprint_sha256": self.dataset_fingerprint,
            },
            "timing_ms": {
                "preprocessing": round(self.preprocessing_ms, 2),
                "inference": round(self.inference_ms, 2),
            },
            "confidence_note": (
                "Softmax model confidence. Confidence is not calibrated and is not a "
                "measure of diagnostic certainty."
            ),
        }


class Predictor:
    """Loads a checkpoint once and performs deterministic inference."""

    def __init__(self, cfg: Config, checkpoint_path: Path | None = None) -> None:
        self.cfg = cfg
        self.checkpoint_path = Path(checkpoint_path or cfg.paths.best_checkpoint)
        self._model = None
        self._classes: list[str] = []
        self._transform = None
        self._device = None
        self._load_seconds = 0.0
        self._meta: dict = {}

    # ── loading ────────────────────────────────────────────────────────────
    def load(self) -> "Predictor":
        """Build and load the model. Idempotent; safe to call from app startup."""
        if self._model is not None:
            return self

        started = time.perf_counter()
        if not self.checkpoint_path.is_file():
            raise CheckpointError(
                f"No model checkpoint at {self.checkpoint_path}.\n"
                "The model has not been trained yet. Train it first:\n"
                "    python grape.py"
            )

        self._device = get_device()

        # Read metadata first so the architecture is built with the *checkpoint's*
        # geometry, not whatever the current config says. Compatibility is
        # deliberately not checked against `cfg` here: inference should follow
        # the checkpoint, and the CLI/API surface a clear error at load time if
        # the classes disagree.
        payload = load_checkpoint(
            self.checkpoint_path,
            device=str(self._device),
            expected_classes=FINAL_CLASSES,
        )
        self._classes = list(payload["classes"])
        stored_cfg = payload.get("config") or {}
        model_cfg = stored_cfg.get("model", {}) if isinstance(stored_cfg, dict) else {}
        image_size = int(payload.get("image_size", self.cfg.train.image_size))

        from iwnet.model.architecture import IWNET

        model = IWNET(
            num_classes=len(self._classes),
            D=int(payload.get("D", model_cfg.get("embedding_dim", 128))),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            bottleneck_dim=int(model_cfg.get("bottleneck_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.3)),
            backbone=str(payload.get("backbone", self.cfg.model.backbone)),
            out_indices=tuple(payload.get("out_indices", self.cfg.model.out_indices)),
            image_size=image_size,
            # Every parameter is overwritten by state_dict below, so downloading
            # ImageNet weights would be pure waste - and it breaks offline use.
            pretrained=False,
            freeze_backbone=False,
        )
        model.load_state_dict(payload["model_state"])
        model.eval()
        model.to(self._device)

        from iwnet.data.dataset import build_eval_transform

        checkpoint_cfg = self._config_from_checkpoint(payload)
        checkpoint_cfg.train.image_size = image_size
        self._transform = build_eval_transform(checkpoint_cfg)

        self._model = model
        self._meta = dict(payload.get("meta", {}))
        self._load_seconds = time.perf_counter() - started
        log.info(
            "Model loaded in %.2fs: %d classes, image_size=%d, device=%s",
            self._load_seconds, len(self._classes), image_size, self._device,
        )
        return self

    @staticmethod
    def _config_from_checkpoint(payload: dict) -> Config:
        """Rebuild a Config from the checkpoint so preprocessing matches training."""
        from iwnet.config import load_config

        cfg = load_config()
        stored = payload.get("config")
        if isinstance(stored, dict):
            from iwnet.config import _merge

            try:
                _merge(cfg, stored)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Could not apply stored config (%s); using current config", exc)
        return cfg

    # ── properties ─────────────────────────────────────────────────────────
    @property
    def classes(self) -> list[str]:
        self._require_loaded()
        return list(self._classes)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    @property
    def image_size(self) -> int:
        """The input resolution this checkpoint was trained at."""
        return int(self._meta.get("image_size", self.cfg.train.image_size))

    @property
    def metadata(self) -> dict:
        return dict(self._meta)

    def _require_loaded(self) -> None:
        if self._model is None:
            raise InferenceError("Predictor.load() has not been called.")

    # ── prediction ─────────────────────────────────────────────────────────
    def predict_pil(self, image) -> Prediction:
        """Run inference on an already-opened RGB PIL image."""
        import torch

        self._require_loaded()
        if image is None:
            raise InferenceError("No image supplied.")

        preprocess_started = time.perf_counter()
        try:
            tensor = self._transform(image).unsqueeze(0)
        except Exception as exc:
            raise InferenceError(f"Preprocessing failed: {exc}") from exc
        preprocessing_ms = (time.perf_counter() - preprocess_started) * 1000

        tensor = tensor.to(self._device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                logits = self._model(tensor)
                probabilities = torch.softmax(logits.float(), dim=1)[0]
        except torch.cuda.OutOfMemoryError as exc:
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
            raise InferenceError(
                "The model ran out of memory handling this image. "
                "Use a smaller image or run inference on CPU."
            ) from exc
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
        inference_ms = (time.perf_counter() - started) * 1000

        scores = probabilities.detach().cpu().numpy()
        best = int(scores.argmax())

        return Prediction(
            predicted_class=self._classes[best],
            confidence=float(scores[best]),
            probabilities={name: float(scores[i]) for i, name in enumerate(self._classes)},
            image_size=int(self._meta.get("image_size", self.cfg.train.image_size)),
            inference_ms=inference_ms,
            preprocessing_ms=preprocessing_ms,
            is_rejection=is_rejection_class(self._classes[best]),
            class_info=describe_class(self._classes[best]),
            model_trained_at=self._meta.get("trained_at_utc"),
            dataset_fingerprint=self._meta.get("dataset_fingerprint_sha256"),
        )

    def predict_path(self, image_path: str | Path) -> Prediction:
        from PIL import Image, UnidentifiedImageError

        path = Path(image_path)
        if not path.is_file():
            raise InferenceError(f"Image not found: {path.name}")
        try:
            with Image.open(path) as handle:
                rgb = handle.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InferenceError(f"Could not read image: {exc}") from exc
        return self.predict_pil(rgb)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS-WIDE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
_PREDICTOR: Predictor | None = None


def get_predictor(cfg: Config | None = None, checkpoint_path: Path | None = None) -> Predictor:
    """Return the cached predictor, loading it on first use."""
    global _PREDICTOR
    if _PREDICTOR is None:
        if cfg is None:
            from iwnet.config import load_config

            cfg = load_config()
        _PREDICTOR = Predictor(cfg, checkpoint_path).load()
    return _PREDICTOR


def reset_predictor() -> None:
    """Drop the cached predictor. Used by tests and by ``--reload``-style flows."""
    global _PREDICTOR
    _PREDICTOR = None


def predict_image_path(image_path: str | Path, cfg: Config | None = None) -> Prediction:
    """Convenience one-shot prediction (still reuses the cached model)."""
    return get_predictor(cfg).predict_path(image_path)
