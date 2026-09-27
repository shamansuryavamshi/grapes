"""Central configuration for the whole project.

Every tunable value lives here. `config.yaml` may override the defaults, and CLI
flags may override `config.yaml`. Nothing else in the codebase hard-codes a
hyperparameter, a path, or a magic number.

RESEARCH-METHODOLOGY NOTE
-------------------------
The defaults below reproduce the hyperparameter values of the original
monolithic `grape.py` exactly (EPOCHS=40, BATCH_SIZE=8, IMG_SIZE=224, LR=1e-3,
UNFREEZE_EPOCH=10, UNFREEZE_STAGES=3, EARLY_STOP_PATIENCE=10, MIXUP_ALPHA=0.3,
MIXUP_PROB=0.5, WARMUP_EPOCHS=3, D=128, VAL_SPLIT=0.15, SEED=42). Changing them
changes experimental results; see README "Research methodology".
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "PROJECT_ROOT",
    "FINAL_CLASSES",
    "ARCHITECTURE",
    "BACKBONE",
    "OUT_INDICES",
    "PathsConfig",
    "DataConfig",
    "AugmentConfig",
    "ModelConfig",
    "TrainConfig",
    "LeakageConfig",
    "ApiConfig",
    "Config",
    "load_config",
    "apply_overrides",
    "SEED",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The 7-class grape-leaf problem. Order is authoritative and is mirrored into
#: every checkpoint so training / validation / inference always agree.
#: NOTE: "Irrelavant" is the project's intentional (misspelled) rejection-class
#: name. It is preserved verbatim for checkpoint and report compatibility.
FINAL_CLASSES: list[str] = [
    "BacterialSpot",
    "Black_Rot",
    "DownyMildew",
    "Esca",
    "Healthy",
    "Irrelavant",
    "PowderyMildew",
]

ARCHITECTURE = "IWNET"

#: The single source of truth for the random seed. The original study used 42 and
#: the reported numbers are only comparable to other runs that share it, so it is
#: defined once here and every stochastic step derives from ``cfg.data.seed``.
SEED: int = 42
BACKBONE = "efficientnet_b3"
#: Feature-map levels tapped from the backbone (stages 2/3/4 of EfficientNet-B3).
OUT_INDICES: tuple[int, int, int] = (1, 2, 3)

IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PathsConfig:
    """Filesystem layout. All paths are resolved against PROJECT_ROOT."""

    source_classes: Path = PROJECT_ROOT / "Balanced_Final_Split"
    source_irrelevant: Path = PROJECT_ROOT / "Irrelevant"
    dataset_root: Path = PROJECT_ROOT / "Balanced_From_Sources"
    config_file: Path = PROJECT_ROOT / "config.yaml"

    @property
    def train_dir(self) -> Path:
        return self.dataset_root / "train"

    @property
    def test_dir(self) -> Path:
        return self.dataset_root / "test"

    @property
    def results_dir(self) -> Path:
        return self.dataset_root / "results"

    @property
    def best_checkpoint(self) -> Path:
        return self.dataset_root / "best_iwnet.pth"

    @property
    def last_checkpoint(self) -> Path:
        return self.dataset_root / "last_iwnet.pth"

    @property
    def frontend_dir(self) -> Path:
        return PROJECT_ROOT / "frontend"


@dataclass
class DataConfig:
    """Dataset construction and loading."""

    seed: int = SEED
    test_split: float = 0.20
    val_split: float = 0.15
    img_exts: tuple[str, ...] = IMG_EXTS
    num_workers: int = 4
    #: ``True``  -> 15% of TRAIN carved out for validation (train-only).
    #: ``False`` -> the 20% test split doubles as the validation set.
    val_from_train: bool = True
    #: Offline minority augmentation (opt-in, TRAIN only).
    aug_enabled: bool = False
    aug_prefix: str = "aug_"
    #: ``None`` -> balance every class up to the largest class.
    aug_target: int | None = None
    #: Raise if any image fails to decode, instead of crashing mid-training.
    fail_on_corrupt: bool = True
    #: Keep visually near-identical images in the SAME split.
    #:
    #: The raw sources are not fully independent samples: they contain
    #: re-encoded and lightly augmented copies of the same photograph, which are
    #: distinct files (so SHA-256 dedup misses them) yet near-identical once
    #: decoded. A plain random split can put such copies on both sides of the
    #: train/test boundary, which inflates the test accuracy.
    #:
    #: With this enabled, images are clustered by perceptual hash and whole
    #: clusters are assigned to one split, so a measured test accuracy reflects
    #: generalisation to genuinely unseen images.
    group_splits: bool = True
    #: Perceptual-hash algorithm for clustering. ``phash`` (DCT) is the default
    #: because it tracks structure rather than brightness: on this dataset
    #: ``ahash`` at a comparable threshold merges hundreds of *unrelated* leaves
    #: into single clusters, which would wreck the split.
    group_algorithm: str = "phash"
    #: pHash Hamming distance at which two images join the same group.
    #:
    #: Calibrated on this dataset: unrelated same-class pairs essentially never
    #: fall within distance 10 (0.01% do), while genuine re-encoded copies land
    #: at distance 0-2. 6 is comfortably above the noise floor and still keeps
    #: the largest cluster small, so the split stays balanced.
    group_threshold: int = 6


@dataclass
class AugmentConfig:
    """Online augmentation strengths (TRAIN ONLY) + offline augmentation ops.

    Every default here is the *original* value from the monolithic ``grape.py``.
    Making them explicit is an engineering change, not a methodology change -
    the augmentation applied to the first training run is identical.

    Grape-leaf relevance notes (see README "Augmentation audit"):
      * Horizontal flip / vertical flip - leaf orientation in a photo is
        arbitrary, so both are safe.
      * RandomResizedCrop(0.5-1.0) - moderately aggressive; can crop out lesions.
        Lower ``scale_min`` to 0.7-0.8 to keep more context.
      * Rotation 45 deg + perspective 0.25 + affine together is a *lot* of
        geometric distortion. Acceptable for background robustness, but it is the
        single most likely cause of destroying small lesion morphology. Reduce
        ``rotation_degrees``/``perspective_p`` if per-class recall suffers.
      * ColorJitter - reasonable proxy for field lighting differences.
      * Grayscale p=0.05 - rare, models illumination-only capture conditions.
      * Sharpness / blur - proxy for focus and camera differences.
      * CutOut / RandomErasing / GaussianNoise - regularisation; mild at these
        settings.
    """

    # Geometric
    crop_scale_min: float = 0.5
    crop_ratio_min: float = 0.75
    crop_ratio_max: float = 1.33
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotation_degrees: int = 45
    affine_translate: float = 0.1
    affine_scale_min: float = 0.85
    affine_scale_max: float = 1.15
    affine_shear: int = 10
    perspective_p: float = 0.3
    perspective_scale: float = 0.25

    # Photometric (PIL, applied before ToTensor)
    jitter_brightness: float = 0.4
    jitter_contrast: float = 0.4
    jitter_saturation: float = 0.4
    jitter_hue: float = 0.15
    grayscale_p: float = 0.05
    sharpness_p: float = 0.3
    blur_p: float = 0.2

    # Tensor space (applied after Normalize)
    gaussian_noise_std: float = 0.01
    cutout_holes: int = 2
    cutout_length: int = 32
    random_erasing_p: float = 0.2
    random_erasing_scale_min: float = 0.02
    random_erasing_scale_max: float = 0.15

    # Offline minority augmentation operators (see iwnet.data.augment)
    offline_rotation_degrees: int = 30
    offline_crop_min: float = 0.85


@dataclass
class ModelConfig:
    """IWNET architecture. Research methodology - do not change casually."""

    backbone: str = BACKBONE
    out_indices: tuple[int, int, int] = OUT_INDICES
    #: Fused embedding width (the ``D`` of the original script).
    embedding_dim: int = 128
    hidden_dim: int = 128
    bottleneck_dim: int = 64
    dropout: float = 0.5
    dropout_hidden: float = 0.3
    num_classes: int = len(FINAL_CLASSES)
    pretrained: bool = True
    #: Freeze the backbone entirely (BN included) until ``unfreeze_epoch``.
    freeze_backbone: bool = True


@dataclass
class TrainConfig:
    """Optimisation. Values mirror the original script unless noted."""

    epochs: int = 40
    batch_size: int = 8
    image_size: int = 224
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    unfreeze_epoch: int = 10
    unfreeze_stages: int = 3
    early_stop_patience: int = 10
    mixup_alpha: float = 0.3
    mixup_probability: float = 0.5
    grad_clip: float = 1.0
    #: New in this version. The original added the freshly unfrozen backbone at
    #: ``LR * 0.01`` but the scheduler overwrote every param group on the very
    #: next step, so the intended lower backbone LR never took effect. This
    #: multiplier is now actually honoured.
    backbone_lr_scale: float = 0.01
    amp: bool = True
    #: Accumulate this many batches per optimiser step (1 = no accumulation).
    grad_accum_steps: int = 1
    use_class_weights: bool = True
    deterministic: bool = True
    log_every_n_batches: int = 5


@dataclass
class LeakageConfig:
    """Duplicate auditing thresholds."""

    #: Perceptual-hash algorithm used for the near-duplicate audit.
    #: ``phash`` is the default: ``ahash`` is dominated by overall brightness, so
    #: on leaf photographs it flags large numbers of unrelated images as
    #: "near-duplicates" and its counts cannot be trusted.
    hash_algorithm: str = "phash"
    #: Hamming distance on a 64-bit perceptual hash at or below which two images
    #: are flagged as near-duplicates. Heuristic only - never a security control.
    near_dup_distance: int = 6
    ahash_size: int = 8
    #: Hard-stop training when exact cross-split duplicates are found.
    strict: bool = True


@dataclass
class ApiConfig:
    """Application server settings. Secrets come from the environment."""

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: float = 10.0
    max_pixels: int = 40_000_000
    cors_origins: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 30
    request_timeout_seconds: int = 30
    serve_frontend: bool = True
    log_level: str = "INFO"


@dataclass
class Config:
    """Top-level configuration object."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))

    def fingerprint(self) -> dict[str, Any]:
        """The subset of config that must match for a checkpoint to be valid."""
        return {
            "architecture": ARCHITECTURE,
            "backbone": self.model.backbone,
            "out_indices": list(self.model.out_indices),
            "embedding_dim": self.model.embedding_dim,
            "hidden_dim": self.model.hidden_dim,
            "bottleneck_dim": self.model.bottleneck_dim,
            "num_classes": self.model.num_classes,
            "image_size": self.train.image_size,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  SERIALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses / Paths / tuples into YAML-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _coerce(value: Any, template: Any) -> Any:
    """Coerce a YAML/CLI value to the type of the dataclass field it targets."""
    if isinstance(template, Path):
        return Path(str(value))
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, tuple):
        if isinstance(value, str):
            value = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(value)
    if isinstance(template, list):
        if isinstance(value, str):
            value = [p.strip() for p in value.split(",") if p.strip()]
        return list(value)
    if template is None:
        return value
    return value


def _merge(base: Any, incoming: dict[str, Any]) -> Any:
    """Recursively merge ``incoming`` into a dataclass instance ``base``."""
    for f in fields(base):
        if f.name not in incoming:
            continue
        current = getattr(base, f.name)
        new_value = incoming[f.name]
        if is_dataclass(current) and isinstance(new_value, dict):
            _merge(current, new_value)
        else:
            setattr(base, f.name, _coerce(new_value, current))
    return base


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def load_config(
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build a :class:`Config` from defaults, then ``config.yaml``, then overrides.

    Overrides use dotted keys, e.g. ``{"train.epochs": 5, "paths.dataset_root": "..."}``.
    A ``config.yaml`` that does not exist is not an error - the defaults are used.
    """
    cfg = Config()

    path = Path(config_file) if config_file else cfg.paths.config_file
    if path.is_file():
        import yaml  # local import: keeps `--help` working without PyYAML

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level")
        _merge(cfg, loaded)

    if overrides:
        _merge(cfg, _nest(overrides))

    return cfg


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{"train.epochs": 5}`` into ``{"train": {"epochs": 5}}``."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def apply_overrides(cfg: Config, dotted: Iterable[str]) -> Config:
    """Apply ``KEY=VALUE`` strings (already known) onto ``cfg`` in place."""
    pairs = {k: v for k, v in (item.split("=", 1) for item in dotted if "=" in item)}
    if pairs:
        _merge(cfg, _nest(pairs))
    return cfg


def config_from_cli_args(args: Any) -> Config:
    """Build a Config from parsed argparse results.

    Only ``None`` CLI values are applied, so unspecified flags never clobber
    ``config.yaml`` settings.
    """
    cfg = load_config(getattr(args, "config", None))

    simple = {
        "epochs": ("train", "epochs", int),
        "batch_size": ("train", "batch_size", int),
        "image_size": ("train", "image_size", int),
        "learning_rate": ("train", "learning_rate", float),
        "weight_decay": ("train", "weight_decay", float),
        "warmup_epochs": ("train", "warmup_epochs", int),
        "unfreeze_epoch": ("train", "unfreeze_epoch", int),
        "unfreeze_stages": ("train", "unfreeze_stages", int),
        "early_stop_patience": ("train", "early_stop_patience", int),
        "mixup_alpha": ("train", "mixup_alpha", float),
        "mixup_prob": ("train", "mixup_probability", float),
        "label_smoothing": ("train", "label_smoothing", float),
        "num_workers": ("data", "num_workers", int),
        "val_split": ("data", "val_split", float),
        "test_split": ("data", "test_split", float),
        "seed": ("data", "seed", int),
        "aug_target": ("data", "aug_target", int),
    }
    for cli_name, (section, field_name, caster) in simple.items():
        value = getattr(args, cli_name, None)
        if value is not None:
            setattr(getattr(cfg, section), field_name, caster(value))

    for flag, field_name in (("no_amp", "amp"), ("no_pretrained", "pretrained")):
        if getattr(args, flag, False):
            section = cfg.train if field_name == "amp" else cfg.model
            setattr(section, field_name, False)

    return cfg


def deep_copy_config(cfg: Config) -> Config:
    """Independent copy, used to freeze run metadata."""
    return copy.deepcopy(cfg)
