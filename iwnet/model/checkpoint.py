"""Checkpoint creation, validation and loading.

A checkpoint is only useful if it can be proven to belong to the same model and
the same data pipeline. :func:`load_checkpoint` therefore refuses to load
mismatched weights instead of discovering the problem as a confusing
``RuntimeError`` from ``load_state_dict`` several frames later - or, worse,
silently accepting a model trained on a different class set.

Security: checkpoints are loaded with ``weights_only=True``. A ``.pth`` file is
an untrusted input; ``weights_only=True`` prevents it from executing arbitrary
pickle payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iwnet.config import ARCHITECTURE, Config
from iwnet.utils import get_logger, utc_timestamp

__all__ = [
    "CheckpointError",
    "CheckpointMeta",
    "build_metadata",
    "save_checkpoint",
    "load_checkpoint",
    "inspect_checkpoint",
]

log = get_logger("model.checkpoint")

REQUIRED_KEYS = ("model_state", "classes", "architecture", "backbone", "epoch", "val_acc", "D")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unreadable, or incompatible."""


@dataclass
class CheckpointMeta:
    """Everything recorded about how a model was produced."""

    architecture: str
    backbone: str
    out_indices: list[int]
    classes: list[str]
    image_size: int
    embedding_dim: int
    num_classes: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mixup_alpha: float
    mixup_probability: float
    label_smoothing: float
    warmup_epochs: int
    unfreeze_epoch: int
    unfreeze_stages: int
    early_stop_patience: int
    dataset_fingerprint: str | None
    built_at_utc: str
    trained_at_utc: str
    git_commit: str | None
    python_version: str
    torch_version: str
    torchvision_version: str
    timm_version: str
    device: str
    epoch: int
    val_acc: float
    train_acc: float | None
    val_loss: float | None
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def build_metadata(
    cfg: Config,
    *,
    epoch: int,
    val_acc: float,
    dataset_fingerprint: str | None,
    device: str,
    train_acc: float | None = None,
    val_loss: float | None = None,
) -> dict[str, Any]:
    """Assemble the reproducibility record stored with every checkpoint."""
    import sys

    from iwnet.utils import git_commit

    report_path = cfg.paths.results_dir / "dataset_report.csv"
    built_at = ""
    if report_path.is_file():
        import csv

        with open(report_path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("metric") == "built_at_utc":
                    built_at = row.get("value", "")
                    break

    return {
        "architecture": ARCHITECTURE,
        "backbone": cfg.model.backbone,
        "out_indices": list(cfg.model.out_indices),
        # Filled in by save_checkpoint(), which owns the authoritative class list.
        "classes": [],
        "image_size": cfg.train.image_size,
        "embedding_dim": cfg.model.embedding_dim,
        "num_classes": cfg.model.num_classes,
        "seed": cfg.data.seed,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "mixup_alpha": cfg.train.mixup_alpha,
        "mixup_probability": cfg.train.mixup_probability,
        "label_smoothing": cfg.train.label_smoothing,
        "warmup_epochs": cfg.train.warmup_epochs,
        "unfreeze_epoch": cfg.train.unfreeze_epoch,
        "unfreeze_stages": cfg.train.unfreeze_stages,
        "early_stop_patience": cfg.train.early_stop_patience,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "dataset_built_at_utc": built_at,
        "trained_at_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "python_version": sys.version.split()[0],
        "torch_version": _pkg_version("torch"),
        "torchvision_version": _pkg_version("torchvision"),
        "timm_version": _pkg_version("timm"),
        "device": device,
        "epoch": epoch,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "config": cfg.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    path: str | Path,
    model_state: dict,
    classes: list[str],
    cfg: Config,
    *,
    epoch: int,
    val_acc: float,
    dataset_fingerprint: str | None = None,
    device: str = "cpu",
    train_acc: float | None = None,
    val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
) -> Path:
    """Write a self-describing checkpoint. Creates parent directories."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(
        cfg,
        epoch=epoch,
        val_acc=val_acc,
        dataset_fingerprint=dataset_fingerprint,
        device=device,
        train_acc=train_acc,
        val_loss=val_loss,
    )
    # The real class list wins over the placeholder in build_metadata.
    meta["classes"] = list(classes)
    if extra:
        meta["extra"] = {**meta.get("extra", {}), **extra}

    payload: dict[str, Any] = {
        "model_state": model_state,
        "classes": list(classes),
        "D": cfg.model.embedding_dim,
        "architecture": meta["architecture"],
        "backbone": meta["backbone"],
        "image_size": meta["image_size"],
        "config": cfg.to_dict(),
        "meta": meta,
        **meta,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state

    # Write to a sibling temp file then replace, so an interrupted save can
    # never leave a half-written checkpoint where a good one used to be.
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    log.info("Saved checkpoint -> %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────────────────────
def _torch_load(path: Path, device: str = "cpu") -> dict:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - torch < 1.13 has no weights_only
        log.warning("This PyTorch build has no weights_only; loading without it (less safe).")
        return torch.load(path, map_location=device)
    except Exception as exc:
        raise CheckpointError(
            f"Could not read checkpoint {path}: {exc}\n"
            "Common causes: the file is truncated (an interrupted save), it is not a "
            "PyTorch checkpoint, or it was written by an incompatible torch version. "
            "Delete it and retrain, or restore a known-good copy."
        ) from exc


def inspect_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a checkpoint for metadata inspection without building a model."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(
            f"Checkpoint not found: {path}\n"
            "Train the model first:  python grape.py"
        )
    payload = _torch_load(path, device)

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(
            f"Checkpoint {path} is missing required keys: {missing}\n"
            f"Found: {sorted(payload.keys())}\n"
            "It was probably written by an older/incompatible script. Retrain to regenerate it."
        )
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_classes: list[str] | None = None,
    cfg: Config | None = None,
) -> dict:
    """Load and **validate** a checkpoint.

    Raises :class:`CheckpointError` describing every mismatch it finds, instead
    of loading a model that would produce meaningless predictions.
    """
    payload = inspect_checkpoint(path, device)

    problems: list[str] = []

    architecture = payload.get("architecture", ARCHITECTURE)
    if architecture != ARCHITECTURE:
        problems.append(
            f"architecture: checkpoint is '{architecture}', this code builds '{ARCHITECTURE}'"
        )

    if expected_classes is not None:
        stored = list(payload.get("classes", []))
        if stored != list(expected_classes):
            problems.append(
                "class set/order mismatch\n"
                f"      checkpoint: {stored}\n"
                f"      expected  : {list(expected_classes)}\n"
                "    A checkpoint trained on a different class set cannot be used."
            )

    if cfg is not None:
        for label, stored_value, expected_value in (
            ("backbone", payload.get("backbone"), cfg.model.backbone),
            ("num_classes", payload.get("num_classes"), cfg.model.num_classes),
            ("embedding_dim (D)", payload.get("D", payload.get("embedding_dim")), cfg.model.embedding_dim),
        ):
            if stored_value != expected_value:
                problems.append(f"{label}: checkpoint={stored_value!r}, config={expected_value!r}")

        stored_indices = list(payload.get("out_indices", []))
        if stored_indices and stored_indices != list(cfg.model.out_indices):
            problems.append(
                f"out_indices: checkpoint={stored_indices}, config={list(cfg.model.out_indices)}"
            )

        stored_size = payload.get("image_size")
        if stored_size is not None and stored_size != cfg.train.image_size:
            problems.append(
                f"image_size: checkpoint={stored_size}, config={cfg.train.image_size}. "
                f"Preprocessing geometry differs, so predictions would shift."
            )

    if problems:
        raise CheckpointError(
            f"Checkpoint {path} is incompatible with the current model/configuration:\n"
            + "\n".join(f"    - {p}" for p in problems)
            + "\n\nRetrain, or point at the matching config/checkpoint pair."
        )

    meta = payload.get("meta", {})
    log.info(
        "Loaded checkpoint: epoch=%s val_acc=%.4f classes=%d image_size=%s",
        payload.get("epoch"), float(payload.get("val_acc", 0.0)),
        len(payload.get("classes", [])), payload.get("image_size"),
    )
    return payload
