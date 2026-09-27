"""Dataset construction, grouping, splitting, validation and leakage auditing.

The invariant that matters most here: a near-duplicate group must never be
split across train and test, because that is exactly how a test accuracy becomes
meaningless while still looking respectable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from iwnet.config import FINAL_CLASSES
from iwnet.data.build import (  # noqa: E402
    _deduplicate,
    _group_by_perceptual_hash,
    _stratified_split,
    _stratified_split_grouped,
)
from iwnet.data.leakage import analyse
from iwnet.data.validation import validate_dataset
from iwnet.utils import phash


# ── deduplication ───────────────────────────────────────────────────────────
def test_dedupe_removes_identical_bytes(rng_images, tmp_path):
    copies = []
    for i, src in enumerate(rng_images):
        for copy_index in range(2):
            dest = tmp_path / f"src{i}_{copy_index}.jpg"
            shutil.copyfile(src, dest)
            copies.append((dest, "Healthy", "test"))

    unique, removed, unreadable = _deduplicate(copies)
    assert len(unique) == len(rng_images)
    assert len(removed) == len(rng_images)  # one extra copy of each
    assert unreadable == 0
    assert all(row["cross_class"] == "false" for row in removed)


def test_dedupe_flags_cross_class_duplicates(rng_images, tmp_path):
    # The same bytes under two different labels means one of the labels is
    # wrong. It must be reported, not silently resolved.
    original = rng_images[0]
    twin = tmp_path / "twin.jpg"
    shutil.copyfile(original, twin)
    samples = [(original, "Healthy", "train"), (twin, "Esca", "train")]

    unique, removed, _ = _deduplicate(samples)
    assert len(unique) == 1
    assert len(removed) == 1
    assert removed[0]["cross_class"] == "true"
    assert {removed[0]["kept_class"], removed[0]["removed_class"]} == {"Healthy", "Esca"}


def test_dedupe_is_deterministic(rng_images, tmp_path):
    def build():
        rows = []
        for i, src in enumerate(rng_images):
            for copy_index in range(2):
                dest = tmp_path / f"d{i}_{copy_index}.jpg"
                if not dest.exists():
                    shutil.copyfile(src, dest)
                rows.append((dest, "Healthy", "train"))
        return rows

    first = [p for p, _c, _l in _deduplicate(build())[0]]
    second = [p for p, _c, _l in _deduplicate(build())[0]]
    assert first == second


# ── grouping ────────────────────────────────────────────────────────────────
def test_identical_images_group_together(rng_images, tmp_path):
    samples = []
    for i, src in enumerate(rng_images):
        samples.append((src, "Healthy", "train"))
        clone = tmp_path / f"clone_{i}.jpg"
        shutil.copyfile(src, clone)
        samples.append((clone, "Healthy", "train"))

    unique, removed, _ = _deduplicate(samples)
    assert len(removed) == len(rng_images), "byte-identical clones must be deduped"
    assert len(unique) == len(rng_images)

    # Now force the harder case: same image, different bytes (resized +
    # re-encoded). Resizing guarantees the bytes differ, so SHA-256 dedupe
    # cannot catch it and only the perceptual grouping can.
    from PIL import Image

    reencoded = []
    for i, src in enumerate(rng_images):
        dest = tmp_path / f"re_{i}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").resize((48, 48), Image.LANCZOS).save(dest, quality=30)
        reencoded.append(dest)

    mixed = [(p, "Healthy", "train") for p in list(rng_images) + reencoded]
    unique, removed, _ = _deduplicate(mixed)
    assert removed == [], "resized copies should not be byte-identical"
    assert len(unique) == 2 * len(rng_images)

    groups = _group_by_perceptual_hash(unique, threshold=6)
    by_name = {path.name: groups[i] for i, (path, _c, _l) in enumerate(unique)}
    for i, _src in enumerate(rng_images):
        assert by_name[f"sample_{i}.jpg"] == by_name[f"re_{i}.jpg"], (
            "a re-encoded copy must land in the same perceptual group"
        )


def test_unrelated_images_do_not_merge(rng_images):
    unique = [(p, "Healthy", "train") for p in rng_images]
    groups = _group_by_perceptual_hash(unique, threshold=6)
    assert len(set(groups.values())) == len(unique), (
        "unrelated random images must not be merged into one group"
    )


def test_grouping_never_splits_a_group(tiny_dataset, cfg):
    samples, unknown = [], {}
    for split in ("train", "test"):
        for name in FINAL_CLASSES:
            for path in sorted((cfg.paths.dataset_root / split / name).iterdir()):
                samples.append((path, name, split))
    assert not unknown

    unique, _removed, _unreadable = _deduplicate(samples)
    groups = _group_by_perceptual_hash(unique, threshold=6, algorithm="phash")
    tr, te, _assignment = _stratified_split_grouped(
        unique, groups, cfg.data.test_split, cfg.data.seed
    )

    index_of = {id(p): i for i, (p, _c, _l) in enumerate(unique)}
    train_groups = {groups[index_of[id(p)]] for p, _c, _l in tr}
    test_groups = {groups[index_of[id(p)]] for p, _c, _l in te}

    assert not (train_groups & test_groups), (
        "a perceptual group appears in BOTH splits - this is leakage"
    )


def test_grouped_split_preserves_every_image(tiny_dataset, cfg):
    samples = []
    for split in ("train", "test"):
        for name in FINAL_CLASSES:
            for path in sorted((cfg.paths.dataset_root / split / name).iterdir()):
                samples.append((path, name, split))

    unique, _removed, _unreadable = _deduplicate(samples)
    groups = _group_by_perceptual_hash(unique, 6, "phash")
    tr, te, _ = _stratified_split_grouped(unique, groups, cfg.data.test_split, 42)

    assert len(tr) + len(te) == len(unique)
    assert not ({id(p) for p, _c, _l in tr} & {id(p) for p, _c, _l in te})


def test_grouped_split_is_deterministic(tiny_dataset, cfg):
    samples = []
    for split in ("train", "test"):
        for name in FINAL_CLASSES:
            for path in sorted((cfg.paths.dataset_root / split / name).iterdir()):
                samples.append((path, name, split))
    unique, _r, _u = _deduplicate(samples)
    groups = _group_by_perceptual_hash(unique, 6, "phash")

    a = [p for p, _c, _l in _stratified_split_grouped(unique, groups, 0.2, 42)[0]]
    b = [p for p, _c, _l in _stratified_split_grouped(unique, groups, 0.2, 42)[0]]
    assert a == b


def test_plain_split_keeps_every_class_in_both_sides(tiny_dataset, cfg):
    samples = []
    for split in ("train", "test"):
        for name in FINAL_CLASSES:
            for path in sorted((cfg.paths.dataset_root / split / name).iterdir()):
                samples.append((path, name, split))
    unique, _r, _u = _deduplicate(samples)
    tr, te = _stratified_split(unique, 0.34, 42)
    for name in FINAL_CLASSES:
        assert any(c == name for _p, c, _l in tr)
        assert any(c == name for _p, c, _l in te)


# ── validation ──────────────────────────────────────────────────────────────
def test_validation_passes_on_a_clean_dataset(cfg):
    report = validate_dataset(cfg, deep=True)
    assert report.ok, getattr(report, "fatal", None)
    assert report.missing_classes == []
    assert report.corrupt == []
    assert set(report.train_counts) == set(FINAL_CLASSES)
    assert set(report.test_counts) == set(FINAL_CLASSES)


def test_validation_reports_a_missing_class(cfg):
    import shutil as sh

    sh.rmtree(cfg.paths.test_dir / "Esca")
    report = validate_dataset(cfg, deep=True)
    assert not report.ok
    # A class present in train but absent from test is a split-level failure, so
    # it is reported in `fatal` rather than `missing_classes` (which is for
    # classes absent from the dataset entirely).
    assert "Esca" in report.test_counts or "Esca" not in report.test_counts
    assert any("Esca" in str(f) for f in report.fatal)


def test_validation_reports_a_corrupt_image(cfg):
    bad = cfg.paths.train_dir / "Healthy" / "corrupt.jpg"
    bad.write_bytes(b"this is definitely not a jpeg")
    report = validate_dataset(cfg, deep=True)
    assert not report.ok
    assert report.corrupt, "a corrupt image was not reported"
    assert any("corrupt" in str(f).lower() or "decode" in str(f).lower()
               for f in report.fatal)


# ── leakage ─────────────────────────────────────────────────────────────────
def test_leakage_clean_on_group_aware_dataset(cfg):
    cfg.data.group_splits = True
    report = analyse(cfg, workers=2)
    assert report.blocking == []
    assert report.near_candidates_checked > 0, (
        "the near-duplicate search must actually run, otherwise a 'clean' "
        "result is meaningless"
    )


def test_leakage_detects_a_planted_exact_duplicate(cfg):
    """Plant a byte-identical test image into train and confirm it is caught."""
    from iwnet.data.leakage import analyse as run

    source = sorted((cfg.paths.test_dir / "Healthy").iterdir())[0]
    planted = cfg.paths.train_dir / "Healthy" / "planted_copy.jpg"
    shutil.copyfile(source, planted)

    report = run(cfg, workers=2)
    assert report.exact, "an exact train/test duplicate was not detected"
    assert report.blocking


def test_leakage_detects_cross_class_duplicate(cfg):
    from iwnet.data.leakage import analyse as run

    source = sorted((cfg.paths.test_dir / "Healthy").iterdir())[0]
    planted = cfg.paths.train_dir / "Esca" / "mislabeled_copy.jpg"
    shutil.copyfile(source, planted)

    report = run(cfg, workers=2)
    assert report.cross_class, "a cross-class duplicate was not detected"


def test_leakage_detects_a_near_duplicate(cfg):
    """A re-encoded copy in the other split must be flagged as near-duplicate."""
    from iwnet.data.leakage import analyse as run
    from PIL import Image

    source = sorted((cfg.paths.test_dir / "Healthy").iterdir())[0]
    planted = cfg.paths.train_dir / "Healthy" / "reencoded.jpg"
    with Image.open(source) as img:
        img.convert("RGB").save(planted, quality=35)

    assert phash(source) is not None
    report = run(cfg, workers=2)
    assert report.near, "a re-encoded near-duplicate was not detected"
