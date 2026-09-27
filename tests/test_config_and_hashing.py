"""Class-name normalisation, hashing and the 7-class contract.

These are the highest-value tests in the suite: the misspelled rejection class
and the alias table are load-bearing for checkpoint compatibility.
"""

from __future__ import annotations

import pytest

from iwnet.config import ARCHITECTURE, FINAL_CLASSES, SEED, load_config
from iwnet.utils import (
    UnknownClassError,
    ahash,
    dhash,
    hamming,
    normalize_class_name,
    perceptual_hash,
    phash,
    sha256_file,
)


def test_exactly_seven_classes():
    assert len(FINAL_CLASSES) == 7


def test_rejection_class_spelling_is_preserved():
    # "Irrelavant" is intentionally misspelled. Renaming it would break every
    # existing checkpoint, report and the training class order.
    assert "Irrelavant" in FINAL_CLASSES
    assert "Irrelevant" not in FINAL_CLASSES
    assert "Irrelavant" in FINAL_CLASSES


def test_class_order_is_stable():
    # The index of each name in this list is baked into saved checkpoints, so a
    # reorder would silently reinterpret every model's output.
    assert FINAL_CLASSES == [
        "BacterialSpot",
        "Black_Rot",
        "DownyMildew",
        "Esca",
        "Healthy",
        "Irrelavant",
        "PowderyMildew",
    ]


def test_seed_is_42():
    assert SEED == 42
    assert load_config().data.seed == 42


def test_architecture_id():
    assert ARCHITECTURE == "IWNET"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BacterialSpot", "BacterialSpot"),
        ("Bacterial_spot", "BacterialSpot"),
        ("Bacterial spot", "BacterialSpot"),
        ("Black_Rot", "Black_Rot"),
        ("Black rot", "Black_Rot"),
        ("DownyMildew", "DownyMildew"),
        ("Downy_Mildew", "DownyMildew"),
        ("Esca", "Esca"),
        ("Esca_(Black_Measles)", "Esca"),
        ("Esca_(Black_Measles)", "Esca"),
        ("Healthy", "Healthy"),
        ("healthy", "Healthy"),
        ("___healthy", "Healthy"),
        ("Grape___healthy", "Healthy"),
        ("Grape___Esca_(Black_Measles)", "Esca"),
        ("Irrelavant", "Irrelavant"),
        ("irrelavant", "Irrelavant"),
        ("IRRELAVANT", "Irrelavant"),
        ("irrelevant", "Irrelavant"),
        ("Irrelavant", "Irrelavant"),
        ("PowderyMildew", "PowderyMildew"),
        ("powdery_mildew", "PowderyMildew"),
    ],
)
def test_alias_normalisation(raw, expected):
    assert normalize_class_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Tomato___Septoria_leaf_spot",
        "Apple___Black_rot",
        "Corn___Common_rust",
        "random_folder",
        "",
    ],
)
def test_unknown_classes_are_rejected(raw):
    # Silently mapping an unknown folder to a class is how mislabelled data gets
    # in. It must raise.
    with pytest.raises(UnknownClassError):
        normalize_class_name(raw)


def test_every_class_normalises_to_itself():
    for name in FINAL_CLASSES:
        assert normalize_class_name(name) == name


def test_sha256_matches_hashlib(rng_images):
    import hashlib

    path = rng_images[0]
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_hamming_counts_set_bits():
    assert hamming(0b1011, 0b0000) == 3
    assert hamming(0b1011, 0b1011) == 0
    assert hamming(0, (1 << 64) - 1) == 64


def test_perceptual_hashes_agree_on_identical_files(rng_images):
    # A byte-identical copy must hash identically under every algorithm.
    import shutil

    original = rng_images[0]
    copy = original.parent / "copy.jpg"
    shutil.copyfile(original, copy)

    for fn in (ahash, phash, dhash):
        a, b = fn(original), fn(copy)
        assert a is not None and b is not None
        assert hamming(a, b) == 0


def test_phash_ignores_rescaling(rng_images):
    # pHash is built to be scale-invariant, which is the property that makes it
    # useful for spotting re-encoded copies.
    from PIL import Image

    src = rng_images[0]
    resized = src.parent / "resized.jpg"
    with Image.open(src) as img:
        img.convert("RGB").resize((96, 96), Image.LANCZOS).save(resized, quality=90)

    assert hamming(phash(src), phash(resized)) <= 6


def test_perceptual_hash_dispatch_rejects_unknown(rng_images):
    with pytest.raises(ValueError):
        perceptual_hash(rng_images[0], "nonexistent")


def test_perceptual_hash_returns_none_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    for fn in (ahash, phash, dhash):
        assert fn(missing) is None
