"""Shared pytest fixtures.

Tests must not need the real dataset or a trained checkpoint, so the fixtures
build tiny synthetic images in a temp directory and use a CPU-only config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iwnet.config import FINAL_CLASSES, load_config  # noqa: E402

#: Enough images per class for a stratified split to be meaningful without the
#: suite becoming slow.
PER_CLASS = 6


def _make_image(path: Path, seed: int) -> None:
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path, quality=95)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A minimal but *valid* 7-class dataset tree under a temp directory."""
    import numpy as np
    from PIL import Image

    root = tmp_path / "TinyDataset"
    for split in ("train", "test"):
        for class_index, name in enumerate(FINAL_CLASSES):
            directory = root / split / name
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(PER_CLASS):
                seed = hash((split, name, i)) % (2**31)
                rng = np.random.default_rng(seed)
                array = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
                ext = "png" if i % 2 else "jpg"
                Image.fromarray(array, mode="RGB").save(
                    directory / f"{name}_{i}.{ext}"
                )
    return root


@pytest.fixture
def cfg(tiny_dataset: Path):
    """Config pointed at the synthetic dataset, CPU only, tiny model."""
    config = load_config(None)
    config.paths.dataset_root = tiny_dataset
    config.train.device = "cpu"
    # 64px is the smallest size the EfficientNet stride-32 backbone tolerates:
    # build_model() runs a forward probe at construction time, and below 64px a
    # BatchNorm in the still-training backbone sees a 1x1 spatial map and raises.
    config.train.image_size = 64
    config.train.batch_size = 2
    config.train.epochs = 1
    config.train.warmup_epochs = 0
    config.train.unfreeze_epoch = 1
    config.data.num_workers = 0
    config.data.val_split = 0.34
    config.model.pretrained = False
    config.model.backbone = "efficientnet_b0"
    return config


@pytest.fixture
def rng_images(tmp_path: Path) -> list[Path]:
    """A handful of unrelated RGB images for prediction tests."""
    import numpy as np
    from PIL import Image

    out = []
    for i in range(3):
        rng = np.random.default_rng(1000 + i)
        array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        p = tmp_path / f"sample_{i}.jpg"
        Image.fromarray(array, mode="RGB").save(p, quality=95)
        out.append(p)
    return out


def png_bytes(size=(64, 64), seed=7) -> bytes:
    """A valid PNG upload for the API tests."""
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


# ── API clients ─────────────────────────────────────────────────────────────
# Shared here rather than in test_api.py so other modules can exercise the real
# FastAPI app too.
#
# app.lifespan calls the ``load_config()`` imported into the app module, and the
# predictor is a process-wide singleton, so both must be controlled *before* the
# client starts.


@pytest.fixture
def api_with_model(cfg, monkeypatch):
    """The real app, wired to a genuinely saved tiny checkpoint."""
    from fastapi.testclient import TestClient

    from iwnet.inference.predictor import reset_predictor
    from iwnet.model.architecture import build_model
    from iwnet.model.checkpoint import save_checkpoint
    import iwnet.api.app as app_module

    model = build_model(cfg, pretrained=False)
    save_checkpoint(
        cfg.paths.best_checkpoint,
        model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
        classes=list(FINAL_CLASSES),
        cfg=cfg,
        epoch=1,
        val_acc=0.5,
    )

    monkeypatch.setattr(app_module, "load_config", lambda *a, **k: cfg)
    reset_predictor()
    with TestClient(app_module.app) as client:
        yield client
    reset_predictor()


@pytest.fixture
def real_config():
    """The project's real, unmodified config: real dataset root and checkpoint.

    Both the dataset and the ``.pth`` files are gitignored, so this fixture
    skips on a clean clone or in CI rather than failing. Tests that need real
    data must depend on this instead of ``cfg``, whose dataset is synthetic.
    """
    config = load_config(None)
    root = Path(config.paths.dataset_root)
    if not root.is_dir():
        pytest.skip(f"real dataset not present at {root}")
    if not Path(config.paths.best_checkpoint).is_file():
        pytest.skip(f"real checkpoint not present at {config.paths.best_checkpoint}")
    return config


@pytest.fixture
def real_api_with_model(real_config, monkeypatch):
    """The real app wired to the real config, checkpoint and dataset tree.

    Used to exercise the ``image_path`` prediction path end to end against
    genuine grape-leaf images. CPU-only so it runs anywhere; the checkpoint is
    loaded, never retrained.
    """
    from fastapi.testclient import TestClient

    from iwnet.inference.predictor import reset_predictor
    import iwnet.api.app as app_module

    real_config.train.device = "cpu"

    monkeypatch.setattr(app_module, "load_config", lambda *a, **k: real_config)
    reset_predictor()
    with TestClient(app_module.app) as client:
        yield client
    reset_predictor()


@pytest.fixture
def api_without_model(monkeypatch, tmp_path):
    """The real app with a config pointing at a checkpoint-free dataset."""
    from fastapi.testclient import TestClient

    from iwnet.config import load_config
    from iwnet.inference.predictor import reset_predictor
    import iwnet.api.app as app_module

    empty = load_config(None)
    empty.paths.dataset_root = tmp_path / "NoModelHere"
    empty.paths.dataset_root.mkdir(parents=True, exist_ok=True)
    assert not empty.paths.best_checkpoint.exists()

    monkeypatch.setattr(app_module, "load_config", lambda *a, **k: empty)
    reset_predictor()
    with TestClient(app_module.app) as client:
        yield client
    reset_predictor()
