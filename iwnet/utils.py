"""Cross-cutting helpers: seeding, device selection, environment reporting,
hashing, class-name normalisation and logging.

No heavy or optional imports at module scope, so ``grape.py --help`` and
``--check-environment`` keep working even when part of the stack is missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "UnknownClassError",
    "set_seed",
    "get_device",
    "device_report",
    "environment_report",
    "sha256_file",
    "sha256_bytes",
    "ahash",
    "phash",
    "dhash",
    "perceptual_hash",
    "hamming",
    "normalize_class_name",
    "class_alias_table",
    "iter_images",
    "count_images",
    "get_logger",
    "configure_logging",
    "utc_timestamp",
    "git_commit",
    "format_bytes",
    "disk_free_gb",
    "human_duration",
    "is_frozen",
]

#: Maps every folder spelling we accept onto the 7 canonical class names.
#:
#: The keys are produced by :func:`normalize_class_name`'s canonicalisation
#: (lowercase, spaces -> underscores, strip surrounding underscores). Anything
#: not in this table raises :class:`UnknownClassError` rather than being silently
#: passed through - that silent pass-through was a real bug: source folders named
#: ``Bacterial_spot`` produced a generated dataset that no longer matched
#: ``FINAL_CLASSES`` and training refused to start.
_CLASS_ALIASES: dict[str, str] = {
    "bacterialspot": "BacterialSpot",
    "bacterial_spot": "BacterialSpot",
    "bacterial_leaf_spot": "BacterialSpot",
    "bacterial_spot_": "BacterialSpot",
    "black_rot": "Black_Rot",
    "blackrot": "Black_Rot",
    "black_rot_leaf": "Black_Rot",
    "downymildew": "DownyMildew",
    "downy_mildew": "DownyMildew",
    "esca": "Esca",
    "esca_(black_measles)": "Esca",
    "esca(black_measles)": "Esca",
    "black_measles": "Esca",
    "healthy": "Healthy",
    "grape___healthy": "Healthy",
    "irrelavant": "Irrelavant",
    "irrelevant": "Irrelavant",
    "powderymildew": "PowderyMildew",
    "powdery_mildew": "PowderyMildew",
    "powdery_mildew_leaf": "PowderyMildew",
}

#: Spellings that mean "this whole folder is the rejection class".
_IRRELEVANT_SPELLINGS = frozenset(
    {"irrelavant", "irrelevant", "irrelavant_", "irrelevant_", "non_grape", "junk"}
)


class UnknownClassError(ValueError):
    """Raised when a source folder name cannot be mapped to a final class."""


# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and CUDA.

    ``deterministic=True`` additionally asks cuDNN for deterministic kernels.
    Full bit-exact reproducibility is *not* achievable on GPU (non-deterministic
    atomics in some backward kernels); this narrows the variance, it does not
    eliminate it. See README "Reproducibility".
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard requirement in practice
        pass

    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some ops have no deterministic CUDA kernel. Raising here
        # would make training impossible on those builds; a warning is honest.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):  # pragma: no cover - very old torch
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def get_device(prefer: str = "auto"):
    """Return the best available ``torch.device``. Never requires CUDA."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                "Run 'python grape.py --check-environment' for diagnostics."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> dict[str, Any]:
    """Human-readable CUDA diagnostics, including why CUDA might be unusable."""
    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": 0,
        "device_names": [],
    }
    try:
        info["device_count"] = torch.cuda.device_count()
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["device_names"].append(f"{props.name} ({props.total_memory / 2**30:.1f} GiB)")
    except Exception as exc:  # pragma: no cover - driver-level failure
        info["error"] = str(exc)

    if not info["cuda_available"]:
        info["diagnosis"] = _diagnose_missing_cuda()
    else:
        try:
            torch.zeros(1, device="cuda")
            info["usable"] = True
        except Exception as exc:
            info["usable"] = False
            info["diagnosis"] = (
                f"PyTorch reports CUDA available but a test allocation failed: {exc}. "
                "Usually a driver/ toolkit mismatch - reinstall the matching torch build."
            )
    return info


def _diagnose_missing_cuda() -> str:
    try:
        import torch

        if torch.version.cuda is None:
            return (
                "This is a CPU-only PyTorch build (torch.version.cuda is None). "
                "Install a CUDA build to use the GPU: "
                "pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision"
            )
        probe = torch.cuda.is_available()
        if not probe:
            return (
                f"PyTorch was built against CUDA {torch.version.cuda} but reports no usable "
                "device. Check that the NVIDIA driver is installed and visible to this user."
            )
    except Exception as exc:  # pragma: no cover
        return f"Could not determine CUDA status: {exc}"
    return "CUDA unavailable for an unknown reason."


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover
        return None


def environment_report() -> dict[str, Any]:
    """Versions and capabilities, for ``--check-environment`` and checkpoints."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": _pkg_version("torch"),
        "torchvision": _pkg_version("torchvision"),
        "timm": _pkg_version("timm"),
        "numpy": _pkg_version("numpy"),
        "pillow": _pkg_version("pillow"),
        "scikit-learn": _pkg_version("scikit-learn"),
        "matplotlib": _pkg_version("matplotlib"),
        "pyyaml": _pkg_version("pyyaml"),
        "fastapi": _pkg_version("fastapi"),
        "uvicorn": _pkg_version("uvicorn"),
    }
    report["device"] = device_report()
    report["disk_free_gb"] = disk_free_gb(Path.cwd())
    return report


def disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(str(path))
        return round(usage.free / 2**30, 2)
    except Exception:  # pragma: no cover
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  HASHING
# ─────────────────────────────────────────────────────────────────────────────
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes - the project's exact-duplicate identity.

    SHA-256 (not MD5) is used everywhere exact identity matters. MD5 is not
    cryptographically secure and is deliberately not used in this codebase.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ahash(path: str | Path, size: int = 8) -> int | None:
    """Average-hash aPEL perceptual hash, returned as an int bitmask.

    This is a *similarity heuristic*, not an identity check: it is invariant to
    small edits and rescaling, which is exactly why it is useful for spotting
    near-duplicates and exactly why it must never be used for security.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            small = img.convert("L").resize((size, size), Image.LANCZOS)
            pixels = small.tobytes()
        mean = sum(pixels) / len(pixels)
        bits = 0
        for index, value in enumerate(pixels):
            if value > mean:
                bits |= 1 << index
        return bits
    except Exception:
        # Unreadable / unsupported file: caller treats None as "cannot compare".
        return None


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ─────────────────────────────────────────────────────────────────────────────
#  PERCEPTUAL HASHES
#
#  Why there is more than one: average-hash (aHash) compares each pixel to the
#  image mean, so it encodes overall brightness. Many grape-leaf photographs are
#  green blobs on a similar background, and at a loose threshold aHash collapses
#  hundreds of *different* leaves into one cluster - which silently destroys a
#  train/test split. pHash keeps only the lowest-frequency 2D-DCT coefficients,
#  so it tracks structure rather than brightness, and is far more selective.
#  dHash (horizontal gradients) is a cheap, robust middle ground.
# ─────────────────────────────────────────────────────────────────────────────
def _dct_matrix(n: int):
    """Orthonormal DCT-II matrix, cached per size."""
    cache = getattr(_dct_matrix, "_cache", None)
    if cache is None:
        cache = {}
        _dct_matrix._cache = cache
    if n not in cache:
        import math

        m = [[0.0] * n for _ in range(n)]
        for k in range(n):
            for x in range(n):
                m[k][x] = math.cos(math.pi * (2 * x + 1) * k / (2 * n))
            scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
            for x in range(n):
                m[k][x] *= scale
        cache[n] = m
    return cache[n]


def phash(path: str | Path, hash_size: int = 8, highfreq_factor: int = 4) -> int | None:
    """DCT perceptual hash (pHash), returned as a 64-bit bitmask.

    Standard construction: grayscale -> resize to ``hash_size*highfreq_factor``
    -> 2D DCT-II -> keep the top-left ``hash_size`` block (dropping the DC term
    from the threshold) -> bit per coefficient against the block median.
    """
    try:
        import numpy as np
        from PIL import Image

        side = hash_size * highfreq_factor
        with Image.open(path) as img:
            small = np.asarray(
                img.convert("L").resize((side, side), Image.LANCZOS), dtype=np.float64
            )
        basis = np.asarray(_dct_matrix(side))
        coeffs = basis @ small @ basis.T
        block = coeffs[:hash_size, :hash_size].copy()
        # The DC term only encodes mean brightness, so exclude it from the median.
        median = np.median(block.flatten()[1:])
        bits = (block > median).astype(np.uint8).flatten()
        value = 0
        for index, bit in enumerate(bits):
            if bit:
                value |= 1 << index
        return value
    except Exception:
        return None


def dhash(path: str | Path, size: int = 8) -> int | None:
    """Gradient (difference) hash: compares horizontally adjacent pixels.

    ``size`` is the output hash edge, so a ``(size+1, size)`` grayscale image is
    reduced to ``size*size`` bits.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
            pixels = list(small.tobytes())
        bits = 0
        for row in range(size):
            base = row * (size + 1)
            for col in range(size):
                if pixels[base + col] > pixels[base + col + 1]:
                    bits |= 1 << (row * size + col)
        return bits
    except Exception:
        return None


def perceptual_hash(path: str | Path, algorithm: str = "phash") -> int | None:
    """Dispatch to a perceptual hash by name."""
    if algorithm == "phash":
        return phash(path)
    if algorithm == "dhash":
        return dhash(path)
    if algorithm == "ahash":
        return ahash(path)
    raise ValueError(f"Unknown perceptual hash algorithm: {algorithm!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS NAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _canonical_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").strip("_")


def class_alias_table() -> dict[str, str]:
    return dict(_CLASS_ALIASES)


def normalize_class_name(name: str) -> str:
    """Map a source folder name onto one of the 7 final classes.

    Unknown names raise :class:`UnknownClassError`. The original code returned
    the input unchanged, which let unexpected folder names flow into the
    generated dataset and break the class assertion downstream.
    """
    raw = str(name).strip()
    if not raw:
        raise UnknownClassError("Empty class name")

    key = _canonical_key(raw)
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]

    # Tolerate the double/underscore padded variants PlantVillage ships with,
    # e.g. "Grape___healthy" or "___Black_rot".
    stripped = key.replace("grape_", "").replace("grape__", "").lstrip("_")
    if stripped in _CLASS_ALIASES:
        return _CLASS_ALIASES[stripped]

    raise UnknownClassError(
        f"Cannot map source class {raw!r} to a final class. "
        f"Expected one of the {len(set(_CLASS_ALIASES.values()))} known spellings, "
        f"e.g. BacterialSpot / Bacterial_spot / Healthy / Irrelavant / irrelevant. "
        f"Add an alias to _CLASS_ALIASES in iwnet/config.py if this folder is legitimate."
    )


def is_irrelevant_spelling(name: str) -> bool:
    return _canonical_key(name) in _IRRELEVANT_SPELLINGS


# ─────────────────────────────────────────────────────────────────────────────
#  FILESYSTEM
# ─────────────────────────────────────────────────────────────────────────────
def iter_images(root: Path, exts: Iterable[str]) -> list[Path]:
    """Every image under ``root`` (recursively), sorted for determinism."""
    allowed = {e.lower() for e in exts}
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed),
        key=lambda p: str(p).lower(),
    )


def count_images(root: Path, exts: Iterable[str]) -> int:
    return len(iter_images(root, exts))


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """One stderr handler with a compact format. Safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger("iwnet")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"iwnet.{name}" if not name.startswith("iwnet") else name)


# ─────────────────────────────────────────────────────────────────────────────
#  MISC
# ─────────────────────────────────────────────────────────────────────────────
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo: Path | None = None) -> str | None:
    """Current git commit, or ``None``. Never raises."""
    try:
        cwd = str(repo) if repo else None
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def format_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
