"""Dataset validation: class-set assertions, per-class counts, corrupt-image scan.

Rules enforced here (from the research protocol):
  * the generated dataset must contain exactly the 7 final classes
  * a missing or unexpected class is a hard failure, not a warning
  * unreadable images are REPORTED, never silently dropped
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from iwnet.config import FINAL_CLASSES, Config
from iwnet.utils import get_logger, iter_images

__all__ = [
    "ValidationReport",
    "count_split",
    "count_dataset",
    "find_missing_classes",
    "find_unexpected_classes",
    "scan_corrupt_images",
    "validate_dataset",
    "CORRUPT_REPORT_NAME",
]

log = get_logger("data.validation")

CORRUPT_REPORT_NAME = "corrupt_images.csv"


@dataclass
class ValidationReport:
    """Outcome of validating a generated dataset."""

    dataset_root: Path
    train_counts: dict[str, int] = field(default_factory=dict)
    test_counts: dict[str, int] = field(default_factory=dict)
    missing_classes: list[str] = field(default_factory=list)
    unexpected_classes: list[str] = field(default_factory=list)
    corrupt: list[dict[str, str]] = field(default_factory=list)
    corrupt_report_path: Path | None = None
    fatal: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal

    @property
    def train_total(self) -> int:
        return sum(self.train_counts.values())

    @property
    def test_total(self) -> int:
        return sum(self.test_counts.values())

    def to_rows(self) -> list[tuple[str, int, int]]:
        """(class, train_count, test_count) in canonical order."""
        return [(c, self.train_counts.get(c, 0), self.test_counts.get(c, 0)) for c in FINAL_CLASSES]


# ─────────────────────────────────────────────────────────────────────────────
#  COUNTING
# ─────────────────────────────────────────────────────────────────────────────
def count_split(split_dir: Path, exts) -> dict[str, int]:
    """Images per class folder, recursing into sub-folders.

    The original ``_count_images_recursive`` was named "recursive" but only
    listed the top level of each class directory, silently under-counting any
    nested layout.
    """
    counts: dict[str, int] = {}
    if not split_dir.is_dir():
        return counts
    for class_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        counts[class_dir.name] = len(iter_images(class_dir, exts))
    return counts


def count_dataset(cfg: Config) -> tuple[dict[str, int], dict[str, int]]:
    return count_split(cfg.paths.train_dir, cfg.data.img_exts), count_split(
        cfg.paths.test_dir, cfg.data.img_exts
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS SET
# ─────────────────────────────────────────────────────────────────────────────
def find_missing_classes(found: set[str]) -> list[str]:
    return [c for c in FINAL_CLASSES if c not in found]


def find_unexpected_classes(found: set[str]) -> list[str]:
    return sorted(c for c in found if c not in FINAL_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
#  CORRUPTION SCAN
# ─────────────────────────────────────────────────────────────────────────────
def scan_corrupt_images(root: Path, exts, limit: int | None = None) -> list[dict[str, str]]:
    """Actually decode every image and report the ones that fail.

    ``ImageFolder`` happily lists unreadable files in ``.samples`` and only fails
    later inside ``__getitem__`` - i.e. after a long training run has already
    started. This runs up front instead.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return []

    corrupt: list[dict[str, str]] = []
    for path in iter_images(root, exts):
        if limit is not None and len(corrupt) >= limit:
            break
        try:
            with Image.open(path) as img:
                img.verify()  # structural check, cheap
            with Image.open(path) as img:
                img.convert("RGB")  # forces a full decode
        except Exception as exc:
            corrupt.append(
                {
                    "path": str(path),
                    "split": root.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return corrupt


# ─────────────────────────────────────────────────────────────────────────────
#  FULL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_dataset(cfg: Config, *, deep: bool = True) -> ValidationReport:
    """Validate the generated dataset, optionally decoding every image."""
    train_dir, test_dir = cfg.paths.train_dir, cfg.paths.test_dir
    report = ValidationReport(dataset_root=cfg.paths.dataset_root)

    if not train_dir.is_dir() or not test_dir.is_dir():
        report.fatal.append(
            f"Generated dataset not found at {cfg.paths.dataset_root}. Run: python grape.py --build-dataset"
        )
        return report

    report.train_counts = count_split(train_dir, cfg.data.img_exts)
    report.test_counts = count_split(test_dir, cfg.data.img_exts)

    found = set(report.train_counts) | set(report.test_counts)
    report.missing_classes = find_missing_classes(found)
    report.unexpected_classes = find_unexpected_classes(found)

    if report.missing_classes:
        report.fatal.append(f"Missing classes: {report.missing_classes}")
    if report.unexpected_classes:
        report.fatal.append(f"Unexpected classes: {report.unexpected_classes}")

    # A class present in only one split is a real leakage/robustness hazard.
    only_train = sorted(set(report.train_counts) - set(report.test_counts))
    only_test = sorted(set(report.test_counts) - set(report.train_counts))
    for cls in only_train:
        report.fatal.append(f"Class {cls!r} exists in train but not in test")
    for cls in only_test:
        report.fatal.append(f"Class {cls!r} exists in test but not in train")

    if report.train_total == 0:
        report.fatal.append("Training split is empty")
    if report.test_total == 0:
        report.fatal.append("Test split is empty")

    if deep:
        report.corrupt = scan_corrupt_images(train_dir, cfg.data.img_exts) + scan_corrupt_images(
            test_dir, cfg.data.img_exts
        )
        if report.corrupt:
            cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
            report.corrupt_report_path = cfg.paths.results_dir / CORRUPT_REPORT_NAME
            with open(report.corrupt_report_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "split", "error"])
                writer.writeheader()
                writer.writerows(report.corrupt)
            if cfg.data.fail_on_corrupt:
                report.fatal.append(
                    f"{len(report.corrupt)} unreadable image(s) found - see "
                    f"{report.corrupt_report_path.name}. "
                    f"Remove/repair them, or set data.fail_on_corrupt=false to continue anyway."
                )

    return report


def print_report(report: ValidationReport) -> None:
    """Console rendering used by ``--dataset-info`` and the training entry point."""
    print("\n[DATASET]")
    print(f"  Root  : {report.dataset_root}")
    print(f"  Train : {report.train_dir if hasattr(report, 'train_dir') else report.dataset_root / 'train'}")
    print(f"  Test  : {report.dataset_root / 'test'}")

    print("\n  Per-class counts (train / test / total):")
    print(f"    {'CLASS':<17}{'TRAIN':>8}{'TEST':>8}{'TOTAL':>9}")
    for cls, tr, te in report.to_rows():
        print(f"    {cls:<17}{tr:>8}{te:>8}{tr + te:>9}")
    print(f"    {'TOTAL':<17}{report.train_total:>8}{report.test_total:>8}{report.train_total + report.test_total:>9}")

    if report.corrupt:
        print(f"\n  [WARN] {len(report.corrupt)} unreadable image(s) -> {report.corrupt_report_path}")
    if report.ok:
        print(f"\n  [OK] Exactly {len(FINAL_CLASSES)} classes present, all splits populated.")
    else:
        print("\n  [ERROR] Dataset validation failed:")
        for problem in report.fatal:
            print(f"    - {problem}")
