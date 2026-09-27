"""Datasets, transforms and DataLoaders.

Two invariants this module exists to guarantee:

1. **One index space.** The original built two separate ``ImageFolder``
   instances (one with train transforms, one with eval transforms) and fed
   ``Subset`` indices derived from the first into the second. That only works
   while both scans enumerate files in exactly the same order - and it breaks
   silently the moment a file is added or removed. Here the directory is scanned
   **once**; train/val/test are index views over that one list, so the indices
   cannot drift.

2. **Deterministic evaluation.** MixUp, CutOut, RandomErasing, rotation, random
   crop, colour jitter and noise exist *only* in the training transform. The
   validation and test transforms are a fixed Resize -> CenterCrop -> ToTensor
   -> Normalize.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from iwnet.config import FINAL_CLASSES, Config
from iwnet.utils import get_logger

__all__ = [
    "CutOut",
    "GaussianNoise",
    "RandomSharpness",
    "RandomGaussianBlur",
    "TransformView",
    "SplitInfo",
    "build_train_transform",
    "build_eval_transform",
    "scan_imagefolder",
    "stratified_train_val_split",
    "build_loaders",
    "imagenet_norm",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

log = get_logger("data.dataset")

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def imagenet_norm() -> tuple[list[float], list[float]]:
    return list(IMAGENET_MEAN), list(IMAGENET_STD)


# ─────────────────────────────────────────────────────────────────────────────
#  TENSOR-SPACE AUGMENTATIONS (train only)
# ─────────────────────────────────────────────────────────────────────────────
class CutOut:
    """Zero out square patches. Applied to a *normalised* tensor.

    Zeroing a normalised tensor writes the dataset mean colour, not black - the
    behaviour matches the original implementation and is intentional.
    """

    def __init__(self, n_holes: int = 2, length: int = 32) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        import torch

        h, w = img.shape[1], img.shape[2]
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            half = self.length // 2
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            mask[:, y1:y2, x1:x2] = 0
        return img * mask


class GaussianNoise:
    """Additive Gaussian noise on a normalised tensor."""

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, img):
        import torch

        return img + torch.randn_like(img) * self.std


class RandomSharpness:
    """PIL sharpness jitter - a proxy for focus/camera variation."""

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, img):
        from PIL import ImageEnhance

        if random.random() < self.p:
            return ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.5))
        return img


class RandomGaussianBlur:
    """PIL blur - a proxy for motion blur / defocus."""

    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, img):
        from PIL import ImageFilter

        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        return img


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORM PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
def build_train_transform(cfg: Config) -> Callable:
    """Full training pipeline. Every value comes from ``cfg.augment``."""
    from torchvision import transforms

    a = cfg.augment
    mean, std = imagenet_norm()
    return transforms.Compose(
        [
            # ── geometric ──
            transforms.RandomResizedCrop(
                cfg.train.image_size, scale=(a.crop_scale_min, 1.0),
                ratio=(a.crop_ratio_min, a.crop_ratio_max),
            ),
            transforms.RandomHorizontalFlip(p=a.hflip_p),
            transforms.RandomVerticalFlip(p=a.vflip_p),
            transforms.RandomRotation(a.rotation_degrees),
            transforms.RandomAffine(
                degrees=0,
                translate=(a.affine_translate, a.affine_translate),
                scale=(a.affine_scale_min, a.affine_scale_max),
                shear=a.affine_shear,
            ),
            transforms.RandomPerspective(distortion_scale=a.perspective_scale, p=a.perspective_p),
            # ── photometric (PIL) ──
            transforms.ColorJitter(
                brightness=a.jitter_brightness,
                contrast=a.jitter_contrast,
                saturation=a.jitter_saturation,
                hue=a.jitter_hue,
            ),
            transforms.RandomGrayscale(p=a.grayscale_p),
            RandomSharpness(p=a.sharpness_p),
            RandomGaussianBlur(p=a.blur_p),
            # ── tensor space ──
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            GaussianNoise(std=a.gaussian_noise_std),
            CutOut(n_holes=a.cutout_holes, length=a.cutout_length),
            transforms.RandomErasing(
                p=a.random_erasing_p,
                scale=(a.random_erasing_scale_min, a.random_erasing_scale_max),
                value="random",
            ),
        ]
    )


def build_eval_transform(cfg: Config) -> Callable:
    """Deterministic Resize -> CenterCrop -> ToTensor -> Normalize.

    Used for BOTH validation and test, and mirrored exactly by the inference
    pipeline in :mod:`iwnet.inference.predictor`.
    """
    from torchvision import transforms

    mean, std = imagenet_norm()
    return transforms.Compose(
        [
            transforms.Resize(round(cfg.train.image_size * 1.14)),
            transforms.CenterCrop(cfg.train.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
#  INDEX VIEW
# ─────────────────────────────────────────────────────────────────────────────
class TransformView:
    """A read-only view over ``(path, label)`` pairs with a transform applied.

    Keeps the sample list and the class map identical to whatever produced them,
    so indices are shared safely between train / validation / test.
    """

    def __init__(self, samples: Sequence[tuple[Path, int]], transform: Callable) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unreadable image {path}: {exc}") from exc
        return self.transform(image), label

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]


def scan_imagefolder(root: Path, exts: Sequence[str], classes: Sequence[str]) -> list[tuple[Path, int]]:
    """One deterministic scan of ``root`` into ``(path, label)`` pairs.

    ``classes`` fixes the label order, so class indices depend only on
    :data:`FINAL_CLASSES` and never on filesystem or dict ordering.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {root}")

    class_to_idx = {name: index for index, name in enumerate(classes)}
    allowed = {e.lower() for e in exts}

    present = {p.name for p in root.iterdir() if p.is_dir()}
    missing = [c for c in classes if c not in present]
    extra = sorted(present - set(classes))
    if missing or extra:
        raise ValueError(
            f"Unexpected class set in {root}. "
            f"Missing: {missing or 'none'}. Unexpected: {extra or 'none'}. "
            f"Expected exactly: {list(classes)}. Rebuild with: python grape.py --build-dataset"
        )

    samples: list[tuple[Path, int]] = []
    for name in classes:  # fixed order -> deterministic labels
        class_dir = root / name
        for path in sorted(class_dir.rglob("*"), key=lambda p: str(p).lower()):
            if path.is_file() and path.suffix.lower() in allowed:
                samples.append((path, class_to_idx[name]))
    if not samples:
        raise ValueError(f"No images found under {root}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  STRATIFIED TRAIN/VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SplitInfo:
    train_idx: list[int] = field(default_factory=list)
    val_idx: list[int] = field(default_factory=list)
    #: Classes whose validation subset came out empty, with the reason.
    starved_classes: dict[str, str] = field(default_factory=dict)
    val_counts: dict[str, int] = field(default_factory=dict)
    train_counts: dict[str, int] = field(default_factory=dict)


def stratified_train_val_split(
    samples: Sequence[tuple[Path, int]],
    classes: Sequence[str],
    val_split: float,
    seed: int,
) -> SplitInfo:
    """Deterministic, class-proportion-preserving split of the *train* split.

    The test set is never involved. Classes too small to yield a validation
    sample are reported in ``starved_classes`` rather than silently vanishing -
    the original used ``int(round(n * 0.15))``, which yields 0 for n <= 3 and
    quietly removed those classes from validation.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(samples):
        by_label.setdefault(label, []).append(index)

    info = SplitInfo()
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        n = len(indices)
        n_val = int(round(n * val_split))
        if n_val == 0 and n >= 2:
            # Keep at least one validation sample so the class is not dropped.
            n_val = 1
        if n_val == 0:
            info.starved_classes[classes[label]] = f"only {n} image(s) in train; cannot carve out validation"
            n_val = 0
        if n_val >= n:
            n_val = max(0, n - 1)
        info.val_idx.extend(indices[:n_val])
        info.train_idx.extend(indices[n_val:])

    info.train_idx.sort()
    info.val_idx.sort()

    for label in range(len(classes)):
        info.train_counts[classes[label]] = sum(1 for i in info.train_idx if samples[i][1] == label)
        info.val_counts[classes[label]] = sum(1 for i in info.val_idx if samples[i][1] == label)
    return info


# ─────────────────────────────────────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(cfg: Config, *, pin_memory: bool | None = None) -> tuple:
    """Build train / validation / test loaders and return them with class names.

    Returns ``(train_loader, val_loader, test_loader, classes, split_info)``.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    classes = list(FINAL_CLASSES)

    train_samples = scan_imagefolder(cfg.paths.train_dir, cfg.data.img_exts, classes)
    test_samples = scan_imagefolder(cfg.paths.test_dir, cfg.data.img_exts, classes)

    split = stratified_train_val_split(train_samples, classes, cfg.data.val_split, cfg.data.seed)

    if cfg.data.val_from_train:
        train_dataset = TransformView([train_samples[i] for i in split.train_idx], build_train_transform(cfg))
        val_dataset = TransformView([train_samples[i] for i in split.val_idx], build_eval_transform(cfg))
    else:
        # Documented alternative: the held-out 20% test split doubles as the
        # validation set. Fewer samples for training; test accuracy is then
        # optimistically biased and must not be quoted as a final result.
        train_dataset = TransformView(train_samples, build_train_transform(cfg))
        val_dataset = TransformView(test_samples, build_eval_transform(cfg))
        log.warning(
            "data.val_from_train=false: validation IS the test split. "
            "Test accuracy will be biased and is not a valid final result."
        )

    test_dataset = TransformView(test_samples, build_eval_transform(cfg))

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    workers = max(0, int(cfg.data.num_workers))
    kwargs = dict(
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    # Windows/macOS spawn worker processes; the CLI entry point is guarded with
    # `if __name__ == "__main__"`, which is what makes this safe.
    if workers > 0:
        kwargs["multiprocessing_context"] = "spawn" if __import__("sys").platform == "win32" else None
        if kwargs["multiprocessing_context"] is None:
            del kwargs["multiprocessing_context"]

    generator = torch.Generator()
    generator.manual_seed(cfg.data.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=False,
        generator=generator, **kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False, **kwargs
    )

    if split.starved_classes:
        for cls, reason in split.starved_classes.items():
            log.warning("Class %r has no validation samples (%s)", cls, reason)

    log.info(
        "Classes (%d): %s", len(classes), ", ".join(classes)
    )
    log.info("Loaders -> train=%d val=%d test=%d", len(train_dataset), len(val_dataset), len(test_dataset))

    return train_loader, val_loader, test_loader, classes, split
