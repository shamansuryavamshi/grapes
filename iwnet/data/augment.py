"""Offline minority-class augmentation - TRAIN ONLY, opt-in, idempotent.

Contract
--------
* Never touches ``test/`` or any validation image.
* Only ever reads *real* images: files whose name starts with ``aug_prefix`` are
  never used as a source, so repeated runs cannot compound augmented images
  into augmented images.
* Counts ``real + already-generated`` against the target, so running it twice
  does not double the dataset.
* Off by default (``--augment`` / ``data.aug_enabled``), because it permanently
  writes files into the generated dataset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from iwnet.config import FINAL_CLASSES, Config
from iwnet.utils import get_logger, iter_images

__all__ = ["AugmentResult", "run_offline_augmentation", "plan_augmentation"]

log = get_logger("data.augment")


@dataclass
class AugmentResult:
    class_counts: dict[str, int] = field(default_factory=dict)
    generated: dict[str, int] = field(default_factory=dict)
    target: int = 0
    total_generated: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.total_generated > 0


def _augment_pil(image, cfg: Config):
    """Geometric + photometric jitter for offline generation.

    Deliberately milder than the online pipeline: these files become permanent
    training data, so aggressive distortion would bake in degraded morphology.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    a = cfg.augment
    out = image.copy()

    if random.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    out = out.rotate(
        random.uniform(-a.offline_rotation_degrees, a.offline_rotation_degrees),
        expand=False,
        fillcolor=(255, 255, 255),
    )
    out = ImageEnhance.Brightness(out).enhance(random.uniform(0.8, 1.2))
    out = ImageEnhance.Contrast(out).enhance(random.uniform(0.8, 1.2))
    out = ImageEnhance.Color(out).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.5:
        width, height = out.size
        factor = random.uniform(a.offline_crop_min, 1.0)
        crop_w, crop_h = int(width * factor), int(height * factor)
        if crop_w > 0 and crop_h > 0 and crop_w <= width and crop_h <= height:
            x0 = random.randint(0, width - crop_w)
            y0 = random.randint(0, height - crop_h)
            out = out.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((width, height))

    if random.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return out


def plan_augmentation(cfg: Config) -> tuple[dict[str, int], int]:
    """Current per-class counts and the target to balance up to."""
    train_dir = cfg.paths.train_dir
    counts: dict[str, int] = {}
    for name in FINAL_CLASSES:
        class_dir = train_dir / name
        if not class_dir.is_dir():
            continue
        images = iter_images(class_dir, cfg.data.img_exts)
        counts[name] = sum(1 for p in images if not p.name.startswith(cfg.data.aug_prefix))
    target = cfg.data.aug_target or (max(counts.values()) if counts else 0)
    return counts, target


def run_offline_augmentation(cfg: Config, *, seed: int | None = None) -> AugmentResult:
    """Generate augmented images for minority classes in the TRAIN split only."""
    from PIL import Image

    random.seed(cfg.data.seed if seed is None else seed)

    print("=" * 68)
    print("  OFFLINE MINORITY AUGMENTATION  (train split only)")
    print("=" * 68)

    train_dir = cfg.paths.train_dir
    if not train_dir.is_dir():
        print(f"[SKIP] Train directory not found: {train_dir}\n")
        return AugmentResult()

    real_counts, target = plan_augmentation(cfg)
    if not real_counts:
        print("[SKIP] No classes found.\n")
        return AugmentResult()

    strategy = (
        f"fixed target ({target})" if cfg.data.aug_target
        else f"largest class ({max(real_counts.values())})"
    )
    print(f"  Real per-class counts : {real_counts}")
    print(f"  Target                : {target}  ({strategy})\n")

    result = AugmentResult(class_counts=real_counts, target=target)

    for name in FINAL_CLASSES:
        class_dir = train_dir / name
        if not class_dir.is_dir():
            result.skipped.append(f"{name}: no directory")
            continue

        images = iter_images(class_dir, cfg.data.img_exts)
        real_files = [p for p in images if not p.name.startswith(cfg.data.aug_prefix)]
        existing = [p for p in images if p.name.startswith(cfg.data.aug_prefix)]
        current = len(real_files) + len(existing)
        needed = target - current

        if needed <= 0:
            print(f"  [OK]   {name:<18} {current}/{target}")
            result.generated[name] = 0
            result.class_counts[name] = current
            continue
        if not real_files:
            message = f"{name}: no source images to augment from"
            result.skipped.append(message)
            print(f"  [SKIP] {name:<18} {message}")
            continue

        print(f"  [AUG]  {name:<18} {current}/{target} -> generating {needed}")
        index = len(existing)
        produced = 0
        failures = 0
        while produced < needed:
            source = random.choice(real_files)
            try:
                with Image.open(source) as handle:
                    augmented = _augment_pil(handle.convert("RGB"), cfg)
                index += 1
                augmented.save(
                    class_dir / f"{cfg.data.aug_prefix}{index:05d}_{source.stem}.jpg",
                    quality=95,
                )
                produced += 1
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    log.warning("Could not augment %s: %s", source, exc)
                produced += 1  # do not loop forever on a single bad file
        if failures:
            result.skipped.append(f"{name}: {failures} generation failure(s)")

        result.generated[name] = produced
        result.class_counts[name] = current + produced
        result.total_generated += produced

    print()
    if result.changed:
        print(f"  Generated {result.total_generated} augmented image(s) in the train split.")
        print("  NOTE: rerunning does not compound - already-augmented files are never used as sources.")
        print("  Re-run the leakage check afterwards, since the dataset changed.\n")
    else:
        print("  All classes already at target - nothing to do.\n")
    return result
