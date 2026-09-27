"""Dataset construction: collect -> normalise -> global dedup -> stratified split.

Order is deliberate and is the documented research pipeline:

    SOURCE DATASETS
      -> CLASS NORMALISATION
      -> GLOBAL DEDUPLICATION (SHA-256, BEFORE any split)
      -> STRATIFIED TRAIN/TEST SPLIT
      -> TRAIN-ONLY VALIDATION SPLIT   (see iwnet.data.dataset)

Deduplication happens *before* the split, never after. Deleting leaked test
images post-hoc cannot undo the fact that the split itself was drawn from a
pool that still contained the duplicates.

The original source folders are opened read-only and are never moved, modified
or deleted. Only the generated ``train/`` and ``test/`` directories are cleared
on rebuild - ``results/`` and the checkpoints are deliberately preserved, which
the original code got wrong by deleting the whole dataset root.
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from iwnet.config import FINAL_CLASSES, Config, PROJECT_ROOT
from iwnet.utils import (
    UnknownClassError,
    get_logger,
    iter_images,
    normalize_class_name,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
)

__all__ = [
    "BuildResult",
    "build_dataset_from_sources",
    "find_source_splits",
    "collect_samples",
    "manifest_fingerprint",
    "print_dataset_info",
    "MANIFEST_NAME",
    "REPORT_NAME",
    "DUPLICATES_NAME",
]

log = get_logger("data.build")

MANIFEST_NAME = "split_manifest.csv"
REPORT_NAME = "dataset_report.csv"
DUPLICATES_NAME = "duplicates_removed.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT TYPES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BuildResult:
    """Everything the CLI/reports need to describe a build. No fabricated data."""

    raw_images: int = 0
    duplicates_removed: int = 0
    unreadable: int = 0
    unique_images: int = 0
    train_total: int = 0
    test_total: int = 0
    train_counts: dict[str, int] = field(default_factory=dict)
    test_counts: dict[str, int] = field(default_factory=dict)
    unknown_classes: dict[str, str] = field(default_factory=dict)
    manifest_path: Path | None = None
    report_path: Path | None = None
    fingerprint: str = ""

    @property
    def ok(self) -> bool:
        return not self.unknown_classes


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def find_source_splits(root: Path) -> tuple[Path, Path] | None:
    """Locate ``<root>/train`` and ``<root>/test``, tolerating one nesting level.

    The real layout is ``Balanced_Final_Split/Balanced_Final_Split/{train,test}``.
    """
    if not root.is_dir():
        return None
    candidates = [root, *sorted((p for p in root.iterdir() if p.is_dir()), key=str)]
    for base in candidates:
        train = base / "train"
        test = base / "test"
        if train.is_dir() and test.is_dir():
            return train, test
    return None


def collect_samples(cfg: Config) -> tuple[list[tuple[Path, str, str]], dict[str, str]]:
    """Gather ``(path, canonical_class, source_label)`` from both source roots.

    The ``Irrelevant`` folder is treated as one flat pool mapped entirely onto
    the single ``Irrelavant`` class - never split into multiple classes.
    Unknown folder names are collected and reported rather than silently kept.
    """
    samples: list[tuple[Path, str, str]] = []
    unknown: dict[str, str] = {}

    splits = find_source_splits(cfg.paths.source_classes)
    if splits is None:
        raise FileNotFoundError(
            f"No train/ and test/ pair found under {cfg.paths.source_classes}. "
            f"Looked for <root>/train and <root>/test, and one level deeper."
        )
    src_train, src_test = splits

    for split_dir, split_name in ((src_train, "source_classes/train"), (src_test, "source_classes/test")):
        for class_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            raw_name = class_dir.name
            try:
                canonical = normalize_class_name(raw_name)
            except UnknownClassError as exc:
                unknown[raw_name] = str(exc)
                log.warning("Skipping unrecognised class folder %s: %s", class_dir, exc)
                continue
            label = f"{split_name}/{raw_name}"
            for image in iter_images(class_dir, cfg.data.img_exts):
                samples.append((image, canonical, label))

    if not cfg.paths.source_irrelevant.is_dir():
        raise FileNotFoundError(f"Irrelevant source folder not found: {cfg.paths.source_irrelevant}")

    # Every variant of "irrelevant" collapses to ONE final class.
    for image in iter_images(cfg.paths.source_irrelevant, cfg.data.img_exts):
        samples.append((image, "Irrelavant", "source_irrelevant"))

    log.info("Collected %d images from sources", len(samples))
    return samples, unknown


# ─────────────────────────────────────────────────────────────────────────────
#  DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def _deduplicate(
    samples: list[tuple[Path, str, str]],
) -> tuple[list[tuple[Path, str, str]], list[dict[str, str]], int]:
    """Global SHA-256 dedup. Keeps the first occurrence in a deterministic order."""
    seen: dict[str, tuple[Path, str]] = {}
    unique: list[tuple[Path, str, str]] = []
    removed: list[dict[str, str]] = []
    unreadable = 0

    for path, canonical, label in samples:
        try:
            digest = sha256_file(path)
        except OSError as exc:
            unreadable += 1
            log.warning("Could not read %s: %s", path, exc)
            continue
        if digest in seen:
            kept_path, kept_class = seen[digest]
            removed.append(
                {
                    "sha256": digest,
                    "kept_path": _relpath(kept_path),
                    "kept_class": kept_class,
                    "removed_path": _relpath(path),
                    "removed_class": canonical,
                    "cross_class": str(kept_class != canonical).lower(),
                }
            )
            continue
        seen[digest] = (path, canonical)
        unique.append((path, canonical, label))

    return unique, removed, unreadable


# ─────────────────────────────────────────────────────────────────────────────
#  GROUPING  (prevent near-duplicate leakage across the split boundary)
# ─────────────────────────────────────────────────────────────────────────────
class _UnionFind:
    """Minimal union-find with path compression, for perceptual-hash clusters."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Always attach the larger root to the smaller for stability.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _group_by_perceptual_hash(
    unique: list[tuple[Path, str, str]], threshold: int, algorithm: str = "phash"
) -> dict[int, int]:
    """Map each sample index to a cluster id using perceptual-hash components.

    Two images join the same cluster when their hash Hamming distance is
    <= ``threshold``. Pairwise comparison is blocked per class and chunked, then
    a union-find merges the results into components - so the cost stays close to
    linear instead of O(n^2) over all pairs.
    """
    import numpy as np

    from iwnet.utils import perceptual_hash

    count = len(unique)
    if count == 0:
        return {}

    hashes: list[int | None] = []
    for path, _canonical, _label in unique:
        try:
            hashes.append(perceptual_hash(path, algorithm))
        except (OSError, ValueError) as exc:
            log.warning("Could not hash %s for grouping: %s", path.name, exc)
            hashes.append(None)

    valid = [i for i, h in enumerate(hashes) if h is not None]
    if len(valid) < 2:
        return {i: 0 for i in range(count)}

    matrix = np.array([hashes[i] for i in valid], dtype=np.uint64)
    uf = _UnionFind(len(valid))

    # Only compare images of the same class: a cross-class visual match is a
    # different (and separately reported) problem, and comparing across classes
    # would merge unrelated classes into one group.
    by_class: dict[str, list[int]] = {}
    for pos, sample_index in enumerate(valid):
        by_class.setdefault(unique[sample_index][1], []).append(pos)

    for _canonical, positions in by_class.items():
        if len(positions) < 2:
            continue
        block = matrix[positions]
        n = block.shape[0]
        chunk = 512
        for start in range(0, n, chunk):
            part = block[start : start + chunk]
            distances = np.bitwise_count(
                np.bitwise_xor(part[:, None], block[None, :])
            )
            rows, cols = np.nonzero(distances <= threshold)
            for r, c in zip(rows.tolist(), cols.tolist()):
                # `r` indexes the chunk, so it needs the chunk offset; `c` already
                # indexes the full block and must not be offset again.
                a, b = start + r, c
                if a < b:
                    uf.union(positions[a], positions[b])

    groups: dict[int, int] = {}
    remap: dict[int, int] = {}
    for pos, sample_index in enumerate(valid):
        root = uf.find(pos)
        if root not in remap:
            remap[root] = len(remap)
        groups[sample_index] = remap[root]
    for i in range(count):
        groups.setdefault(i, -1)
    return groups


def _stratified_split_grouped(
    unique: list[tuple[Path, str, str]],
    groups: dict[int, int],
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]], dict[int, int]]:
    """Split whole perceptual-hash groups, keeping each class roughly balanced.

    Groups are shuffled deterministically, then greedily placed into whichever
    split is furthest below its target. Every member of a group follows it, so
    near-identical images can never straddle the boundary.
    """
    by_class: dict[str, dict[int, list[int]]] = {}
    for index, (path, canonical, label) in enumerate(unique):
        group = groups.get(index, -1)
        by_class.setdefault(canonical, {}).setdefault(group, []).append(index)

    rng = random.Random(seed)
    train_items: list[tuple[Path, str, str]] = []
    test_items: list[tuple[Path, str, str]] = []
    assignment: dict[int, int] = {}

    for canonical in sorted(by_class):
        class_groups = by_class[canonical]
        sizes = {g: len(idx) for g, idx in class_groups.items()}
        order = sorted(class_groups)
        rng.shuffle(order)
        # Largest groups first: big groups have fewer valid placements, so
        # placing them while both splits are empty keeps balance tight.
        order.sort(key=lambda g: -sizes[g])

        total = sum(sizes.values())
        target_test = total * test_fraction
        target_train = total - target_test
        have_test = 0
        have_train = 0

        for group in order:
            size = sizes[group]
            # Prefer the split that is furthest below its target.
            deficit_test = target_test - have_test
            deficit_train = target_train - have_train
            if deficit_test <= 0 and deficit_train <= 0:
                to_test = False
            elif deficit_train <= 0:
                to_test = True
            elif deficit_test <= 0:
                to_test = False
            else:
                # Normalised deficit so a large group is not always sent to the
                # split that happens to have the bigger absolute shortfall.
                to_test = (deficit_test / target_test) >= (deficit_train / target_train)

            for index in class_groups[group]:
                item = unique[index]
                (test_items if to_test else train_items).append(item)
                assignment[index] = 1 if to_test else 0
            if to_test:
                have_test += size
            else:
                have_train += size

    return train_items, test_items, assignment


# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def _stratified_split(
    unique: list[tuple[Path, str, str]], test_fraction: float, seed: int
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]]]:
    """Deterministic, class-aware 80/20 split."""
    by_class: dict[str, list[tuple[Path, str, str]]] = {}
    for item in unique:
        by_class.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train_items: list[tuple[Path, str, str]] = []
    test_items: list[tuple[Path, str, str]] = []

    for canonical in sorted(by_class):
        items = sorted(by_class[canonical], key=lambda t: str(t[0]).lower())
        rng.shuffle(items)
        if len(items) <= 1:
            train_items.extend(items)
            continue
        n_test = max(1, round(len(items) * test_fraction))
        n_test = min(n_test, len(items) - 1)  # never empty out a class
        test_items.extend(items[:n_test])
        train_items.extend(items[n_test:])

    return train_items, test_items


# ─────────────────────────────────────────────────────────────────────────────
#  WRITING
# ─────────────────────────────────────────────────────────────────────────────
def _relpath(path: Path) -> str:
    """Project-relative path when possible, else absolute. Never a bare name."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _copy_unique(src: Path, dest_dir: Path) -> Path:
    """Copy into ``dest_dir``, suffixing on collision so nothing is overwritten."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        index = 1
        while (dest_dir / f"{stem}__{index}{suffix}").exists():
            index += 1
        dest = dest_dir / f"{stem}__{index}{suffix}"
    shutil.copy2(src, dest)
    return dest


def _clear_generated_splits(dataset_root: Path) -> None:
    """Remove ONLY the generated train/ and test/ trees.

    The original implementation called ``shutil.rmtree(DS_ROOT)``, which also
    deleted ``results/`` and ``best_iwnet.pth`` - so rebuilding the dataset threw
    away a trained model and every evaluation graph.
    """
    for name in ("train", "test"):
        target = dataset_root / name
        if target.exists():
            log.info("Clearing generated split directory: %s", target)
            shutil.rmtree(target)


def manifest_fingerprint(rows: list[dict[str, str]]) -> str:
    """Deterministic SHA-256 over (split, class, sha256) of every image.

    Lets a checkpoint be tied to the exact dataset it was trained on.
    """
    payload = "\n".join(
        f"{row['split']}|{row['class']}|{row['sha256']}"
        for row in sorted(rows, key=lambda r: (r["split"], r["class"], r["sha256"]))
    )
    return sha256_bytes(payload.encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def _read_manifest_fingerprint(dataset_root: Path) -> str | None:
    """Fingerprint recorded by the last build, so runs can be tied to their data."""
    import csv as _csv

    report = dataset_root / "results" / REPORT_NAME
    if not report.is_file():
        return None
    try:
        with open(report, newline="", encoding="utf-8") as handle:
            for row in _csv.DictReader(handle):
                # The report is written with the columns "metric,value".
                # ("key,value" is accepted too so an older report still reads.)
                if row.get("metric", row.get("key")) == "dataset_fingerprint_sha256":
                    value = row.get("value")
                    if value:
                        return value
    except OSError:
        return None
    return None


def print_dataset_info(cfg: Config) -> None:
    """Print the generated dataset's layout, counts and provenance."""
    root = cfg.paths.dataset_root
    print("=" * 68)
    print("  DATASET INFO")
    print("=" * 68)
    print(f"  Root            : {root}")
    print(f"  Train           : {cfg.paths.train_dir}")
    print(f"  Test            : {cfg.paths.test_dir}")

    if not cfg.paths.train_dir.is_dir():
        print("\n  [MISSING] The generated dataset does not exist yet.")
        print("  Build it with:  python grape.py --build-dataset\n")
        return

    from iwnet.data.validation import count_dataset

    train_counts, test_counts = count_dataset(cfg)

    print()
    print(f"  {'CLASS':<18}{'TRAIN':>8}{'TEST':>8}{'TOTAL':>9}")
    print("  " + "-" * 43)
    train_total = test_total = 0
    for canonical in FINAL_CLASSES:
        tr, te = train_counts.get(canonical, 0), test_counts.get(canonical, 0)
        train_total += tr
        test_total += te
        print(f"  {canonical:<18}{tr:>8}{te:>8}{tr + te:>9}")
    print("  " + "-" * 43)
    print(f"  {'TOTAL':<18}{train_total:>8}{test_total:>8}{train_total + test_total:>9}")

    if train_total:
        print(
            f"\n  Test fraction   : {test_total / (train_total + test_total) * 100:.1f}%"
        )

    print(f"  Group-aware split: {'ON' if cfg.data.group_splits else 'OFF'}"
          f"  ({cfg.data.group_algorithm} <= {cfg.data.group_threshold})")
    print(f"  Seed            : {cfg.data.seed}")

    fingerprint = _read_manifest_fingerprint(root)
    if fingerprint:
        print(f"  Fingerprint     : {fingerprint}")
    print(f"  Manifest        : {cfg.paths.results_dir / MANIFEST_NAME}")
    print(
        f"  Checkpoint      : {cfg.paths.best_checkpoint}"
        f"  ({'present' if cfg.paths.best_checkpoint.is_file() else 'not trained'})"
    )
    print("=" * 68)


def build_dataset_from_sources(
    cfg: Config,
    *,
    test_fraction: float | None = None,
    dry_run: bool = False,
) -> BuildResult:
    """Build ``Balanced_From_Sources`` from the two source folders.

    ``dry_run=True`` performs every step except writing image files, so the
    dataset can be audited without touching the filesystem.
    """
    test_fraction = cfg.data.test_split if test_fraction is None else test_fraction
    root = cfg.paths.dataset_root

    print("=" * 68)
    print("  BUILD DATASET FROM SOURCES")
    print("=" * 68)
    print(f"  Source 1 (classes)    : {cfg.paths.source_classes}")
    print(f"  Source 2 (irrelevant) : {cfg.paths.source_irrelevant}")
    print(f"  Output dataset        : {root}")
    print(f"  Test fraction         : {test_fraction:.2f}   Seed: {cfg.data.seed}\n")

    samples, unknown = collect_samples(cfg)
    if unknown:
        print("[ERROR] Unrecognised source class folders were found:\n")
        for raw, reason in unknown.items():
            print(f"  {raw!r}: {reason}\n")
        print("Refusing to build a dataset with an ambiguous class set.")
        print("Add the correct alias to _CLASS_ALIASES in iwnet/config.py, then rerun.")
        return BuildResult(raw_images=len(samples), unknown_classes=unknown)

    unique, removed, unreadable = _deduplicate(samples)
    print(f"  Collected images (all sources) : {len(samples)}")
    print(f"  Unreadable files skipped        : {unreadable}")
    print(f"  Exact duplicates removed (SHA-256): {len(removed)}")
    print(f"  Unique images                   : {len(unique)}")

    cross_class = sum(1 for r in removed if r["cross_class"] == "true")
    if cross_class:
        print(f"    of which CROSS-CLASS duplicates : {cross_class}  (same bytes, two labels)")

    # ── Group near-duplicates so they cannot straddle the split boundary ────
    split_assignment: dict[int, int] = {}
    group_of: dict[int, int] = {}
    if cfg.data.group_splits:
        group_of = _group_by_perceptual_hash(
            unique, cfg.data.group_threshold, cfg.data.group_algorithm
        )
        cluster_ids = {g for g in group_of.values() if g >= 0}
        multi = len(unique) - len(cluster_ids)
        print(
            f"\n  Near-duplicate grouping"
            f" ({cfg.data.group_algorithm} <= {cfg.data.group_threshold})"
            f"\n    perceptual groups              : {len(cluster_ids)}"
            f"\n    images in a multi-image group   : {multi}"
        )
        train_items, test_items, split_assignment = _stratified_split_grouped(
            unique, group_of, test_fraction, cfg.data.seed
        )
        print("    Split is group-aware: whole groups go to one side.")
    else:
        print(
            "\n  [WARNING] Group-aware splitting is DISABLED. Near-identical images"
            "\n            can land in both train and test, which inflates test accuracy."
        )
        train_items, test_items = _stratified_split(unique, test_fraction, cfg.data.seed)

    train_counts = {c: 0 for c in FINAL_CLASSES}
    test_counts = {c: 0 for c in FINAL_CLASSES}
    for _, canonical, _ in train_items:
        train_counts[canonical] += 1
    for _, canonical, _ in test_items:
        test_counts[canonical] += 1

    index_of = {id(path): i for i, (path, _c, _l) in enumerate(unique)}
    manifest_rows: list[dict[str, str]] = []
    for split_name, items in (("train", train_items), ("test", test_items)):
        for path, canonical, label in items:
            index = index_of[id(path)]
            group = group_of.get(index, -1)
            manifest_rows.append(
                {
                    "split": split_name,
                    "class": canonical,
                    "filename": path.name,
                    "source": _relpath(path),
                    "source_group": label,
                    "perceptual_group": "n/a" if group < 0 else str(group),
                    "sha256": sha256_file(path),
                }
            )
    fingerprint = manifest_fingerprint(manifest_rows)

    print("\n  Final per-class counts (train / test / total):")
    for canonical in FINAL_CLASSES:
        tr, te = train_counts[canonical], test_counts[canonical]
        print(f"    {canonical:<16} train={tr:<6} test={te:<6} total={tr + te}")
    print(f"    {'TOTAL':<16} train={len(train_items):<6} test={len(test_items):<6} total={len(unique)}")

    missing = [c for c in FINAL_CLASSES if (train_counts[c] + test_counts[c]) == 0]
    if missing:
        print(f"\n[ERROR] These final classes have zero images: {missing}")
        print("Refusing to build. Every one of the 7 classes must be represented.")
        return BuildResult(
            raw_images=len(samples), unique_images=len(unique), unknown_classes={"<empty>": ", ".join(missing)}
        )

    if dry_run:
        print("\n  [DRY RUN] No files written.")
        return BuildResult(
            raw_images=len(samples),
            duplicates_removed=len(removed),
            unreadable=unreadable,
            unique_images=len(unique),
            train_total=len(train_items),
            test_total=len(test_items),
            train_counts=train_counts,
            test_counts=test_counts,
            fingerprint=fingerprint,
        )

    # Write images, clearing ONLY the generated splits.
    _clear_generated_splits(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for split_name, items in (("train", train_items), ("test", test_items)):
        for path, canonical, _ in items:
            _copy_unique(path, root / split_name / canonical)
            written += 1
    print(f"\n  Wrote {written} image files.")

    # ── Manifest ────────────────────────────────────────────────────────────
    manifest_path = cfg.paths.results_dir / MANIFEST_NAME
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "class",
                "filename",
                "source",
                "source_group",
                "perceptual_group",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: (r["split"], r["class"], r["filename"])))

    # ── Duplicate audit trail ───────────────────────────────────────────────
    if removed:
        duplicates_path = cfg.paths.results_dir / DUPLICATES_NAME
        with open(duplicates_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sha256", "kept_path", "kept_class", "removed_path", "removed_class", "cross_class"],
            )
            writer.writeheader()
            writer.writerows(removed)

    # ── Report ──────────────────────────────────────────────────────────────
    report_path = cfg.paths.results_dir / REPORT_NAME
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["built_at_utc", utc_timestamp()])
        writer.writerow(["source_classes", str(cfg.paths.source_classes)])
        writer.writerow(["source_irrelevant", str(cfg.paths.source_irrelevant)])
        writer.writerow(["dataset_root", str(root)])
        writer.writerow(["seed", cfg.data.seed])
        writer.writerow(["test_fraction", f"{test_fraction:.4f}"])
        writer.writerow(["val_split_of_train", f"{cfg.data.val_split:.4f}"])
        writer.writerow(["raw_images", len(samples)])
        writer.writerow(["unreadable_skipped", unreadable])
        writer.writerow(["exact_duplicates_removed", len(removed)])
        writer.writerow(["cross_class_duplicates_removed", cross_class])
        writer.writerow(["unique_images", len(unique)])
        writer.writerow(["train_images", len(train_items)])
        writer.writerow(["test_images", len(test_items)])
        writer.writerow(["dataset_fingerprint_sha256", fingerprint])
        for canonical in FINAL_CLASSES:
            writer.writerow([f"train_{canonical}", train_counts[canonical]])
            writer.writerow([f"test_{canonical}", test_counts[canonical]])

    print(f"  Manifest -> {manifest_path}")
    print(f"  Report   -> {report_path}")
    if removed:
        print(f"  Duplicates -> {cfg.paths.results_dir / DUPLICATES_NAME}")
    print(f"  Dataset fingerprint (SHA-256): {fingerprint}")
    print("\n  Next:  python grape.py --dataset-info\n")

    return BuildResult(
        raw_images=len(samples),
        duplicates_removed=len(removed),
        unreadable=unreadable,
        unique_images=len(unique),
        train_total=len(train_items),
        test_total=len(test_items),
        train_counts=train_counts,
        test_counts=test_counts,
        manifest_path=manifest_path,
        report_path=report_path,
        fingerprint=fingerprint,
    )
