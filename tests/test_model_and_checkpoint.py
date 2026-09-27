"""Architecture, checkpoint round-trip, and metadata-safety.

The important guarantees: a checkpoint must refuse to load against a mismatched
class order or geometry, and inference must never require network access.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
import torch

from iwnet.config import FINAL_CLASSES
from iwnet.model.architecture import build_model, describe_model
from iwnet.model.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from iwnet.training.metrics import ClassificationMetrics, PerClassMetric


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_output_matches_class_count(cfg):
    model = build_model(cfg, pretrained=False)
    model.eval()
    size = cfg.train.image_size
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, size, size))
    assert logits.shape == (2, len(FINAL_CLASSES))


def test_model_tolerates_a_different_input_resolution(cfg):
    # The global-pooling head must accept any spatial size, otherwise inference
    # on a differently-sized upload silently breaks.
    model = build_model(cfg, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 48, 48))
    assert logits.shape == (1, len(FINAL_CLASSES))


def test_describe_model_reports_geometry(cfg):
    text = describe_model(build_model(cfg, pretrained=False)).lower()
    assert str(cfg.train.image_size) in text


def test_freezing_controls_trainable_parameters(cfg):
    cfg.model.freeze_backbone = True
    model = build_model(cfg, pretrained=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < total, "frozen backbone still has trainable parameters"

    cfg.model.freeze_backbone = False
    model = build_model(cfg, pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


def test_frozen_backbone_batchnorm_stays_in_eval(cfg):
    """Frozen BatchNorm must not keep updating running stats.

    ``requires_grad=False`` alone does not stop that; the backbone would drift
    on a small dataset before it is ever trained.
    """
    cfg.model.freeze_backbone = True
    model = build_model(cfg, pretrained=False)
    model.train()
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert not module.training, "a frozen BatchNorm was left in train mode"


def test_checkpoint_round_trip_preserves_predictions(cfg, tmp_path):
    model = build_model(cfg, pretrained=False)
    model.eval()
    size = cfg.train.image_size
    probe = torch.randn(1, 3, size, size)
    with torch.no_grad():
        before = model(probe).clone()

    path = tmp_path / "ckpt.pth"
    save_checkpoint(
        path,
        model_state=_state(model),
        classes=list(FINAL_CLASSES),
        cfg=cfg,
        epoch=3,
        val_acc=0.5,
    )
    assert path.exists()

    payload = load_checkpoint(path, device="cpu",
                             expected_classes=list(FINAL_CLASSES))
    rebuilt = build_model(cfg, pretrained=False)
    rebuilt.load_state_dict(payload["model_state"])
    rebuilt.eval()
    with torch.no_grad():
        after = rebuilt(probe)

    assert torch.allclose(before, after, atol=1e-5), (
        "reloaded weights do not reproduce the original model's output"
    )
    assert payload["epoch"] == 3
    assert payload["classes"] == list(FINAL_CLASSES)
    assert payload["image_size"] == cfg.train.image_size


def test_checkpoint_rejects_class_order_mismatch(cfg, tmp_path):
    path = tmp_path / "bad_classes.pth"
    save_checkpoint(path, model_state=_state(build_model(cfg, pretrained=False)),
                    classes=list(FINAL_CLASSES), cfg=cfg, epoch=1, val_acc=0.1)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["classes"] = list(reversed(FINAL_CLASSES))
    torch.save(payload, path)

    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(path, device="cpu", expected_classes=list(FINAL_CLASSES))
    assert "class" in str(excinfo.value).lower()


def test_checkpoint_rejects_geometry_mismatch(cfg, tmp_path):
    path = tmp_path / "bad_geometry.pth"
    save_checkpoint(path, model_state=_state(build_model(cfg, pretrained=False)),
                    classes=list(FINAL_CLASSES), cfg=cfg, epoch=1, val_acc=0.1)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["image_size"] = 999
    torch.save(payload, path)

    with pytest.raises(CheckpointError):
        load_checkpoint(path, device="cpu", cfg=cfg)


def test_checkpoint_rejects_foreign_architecture(cfg, tmp_path):
    path = tmp_path / "other_arch.pth"
    save_checkpoint(path, model_state=_state(build_model(cfg, pretrained=False)),
                    classes=list(FINAL_CLASSES), cfg=cfg, epoch=1, val_acc=0.1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["architecture"] = "SOMETHING_ELSE"
    torch.save(payload, path)

    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(path, device="cpu", expected_classes=list(FINAL_CLASSES))
    assert "architecture" in str(excinfo.value).lower()


def test_checkpoint_stores_dataset_fingerprint(cfg, tmp_path):
    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, model_state=_state(build_model(cfg, pretrained=False)),
                    classes=list(FINAL_CLASSES), cfg=cfg, epoch=7, val_acc=0.9,
                    dataset_fingerprint="abc123")
    payload = load_checkpoint(path, device="cpu", expected_classes=list(FINAL_CLASSES))
    assert payload["dataset_fingerprint_sha256"] == "abc123"


def test_missing_checkpoint_raises_a_clear_error(tmp_path):
    with pytest.raises((CheckpointError, FileNotFoundError, OSError)):
        load_checkpoint(tmp_path / "nope.pth", device="cpu")


def test_metrics_summary_is_ascii_safe():
    """A Windows cp1252 console must not crash on the summary table.

    This bug previously killed the run *after* training and checkpointing.
    """
    metrics = ClassificationMetrics(
        classes=list(FINAL_CLASSES),
        accuracy=0.9,
        balanced_accuracy=0.9,
        macro_f1=0.9,
        weighted_f1=0.9,
        macro_precision=0.9,
        macro_recall=0.9,
        total_samples=70,
        per_class=[
            PerClassMetric(name=n, precision=0.9, recall=0.9, f1=0.9,
                           auc=0.9, accuracy=0.9, support=10)
            for n in FINAL_CLASSES
        ],
        confusion_matrix=[[0] * 7 for _ in range(7)],
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        metrics.print_summary("TEST")

    text = buf.getvalue()
    text.encode("cp1252")  # raises if a non-cp1252 character slipped in
    assert "SET METRICS" in text
    assert "Irrelavant" in text
