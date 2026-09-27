"""Training-loop details that are easy to get silently wrong.

Covers the LR schedule, MixUp accounting, class weights, and the end-to-end
inference path against a real (tiny) trained checkpoint.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from iwnet.config import FINAL_CLASSES
from iwnet.training.trainer import (
    WarmupCosineScheduler,
    compute_class_weights,
    mixup_criterion,
    mixup_data,
)


def _optimizer(lr: float = 1e-3):
    model = nn.Linear(4, 2)
    return model, torch.optim.AdamW(model.parameters(), lr=lr)


# ── scheduler ───────────────────────────────────────────────────────────────
def test_warmup_raises_the_lr_from_a_nonzero_start():
    """A plain epoch/warmup ramp gives LR 0 for epoch 1 - no training at all."""
    model, opt = _optimizer()
    sched = WarmupCosineScheduler(opt, warmup_epochs=4, total_epochs=20, base_lr=1e-3)

    sched.apply()
    first = opt.param_groups[0]["lr"]
    assert first > 0, "the first epoch must have a non-zero learning rate"

    sched.step()
    sched.apply()
    later = opt.param_groups[0]["lr"]
    assert later > first, "warmup did not increase the learning rate"

    sched.step()
    sched.apply()
    assert opt.param_groups[0]["lr"] > later


def test_schedule_reaches_base_lr_at_the_end_of_warmup():
    model, opt = _optimizer()
    sched = WarmupCosineScheduler(opt, warmup_epochs=3, total_epochs=30, base_lr=1e-3)
    for _ in range(3):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-6)


def test_cosine_decays_monotonically_after_warmup():
    model, opt = _optimizer()
    sched = WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=20, base_lr=1e-3)
    for _ in range(2):
        sched.step()
    lrs = []
    for _ in range(18):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine decay is not monotonic"
    assert lrs[-1] < lrs[0]


def test_final_epoch_decays_to_zero():
    model, opt = _optimizer()
    total = 15
    sched = WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=total, base_lr=1e-3)
    for _ in range(total):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12), (
        "the last epoch should land on exactly 0 so training actually settles"
    )


def test_factor_rises_then_falls_as_one_curve():
    model, opt = _optimizer()
    sched = WarmupCosineScheduler(opt, warmup_epochs=3, total_epochs=25, base_lr=1e-3)
    factors = []
    for _ in range(25):
        factors.append(sched.factor())
        sched.step()
    warm, decay = factors[:3], factors[3:]
    assert all(a <= b + 1e-12 for a, b in zip(warm, warm[1:])), "warmup must rise"
    assert all(a >= b - 1e-12 for a, b in zip(decay, decay[1:])), "cosine must fall"
    assert all(0.0 <= f <= 1.0 + 1e-12 for f in factors)


def test_sync_registers_a_late_param_group_without_rebasing_others():
    """A group added at unfreeze must keep its own (reduced) base LR.

    Re-reading every group's *current* LR here would capture the already-decayed
    value as a new base and collapse the head's LR.
    """
    model, opt = _optimizer()
    sched = WarmupCosineScheduler(opt, warmup_epochs=0, total_epochs=10, base_lr=1e-3)
    for _ in range(5):
        sched.step()
    head_lr_before = opt.param_groups[0]["lr"]

    opt.add_param_group({"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-5})
    sched.sync()

    assert len(sched.base_lrs) == 2
    assert sched.base_lrs[1] == pytest.approx(1e-5)
    sched.step()
    assert opt.param_groups[0]["lr"] < head_lr_before, "the head LR should keep decaying"
    assert opt.param_groups[0]["lr"] > 1e-5, "the head LR must not collapse"


# ── MixUp ───────────────────────────────────────────────────────────────────
def test_mixup_data_keeps_batch_size_and_is_a_convex_blend():
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    y = torch.arange(4)
    blended, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    assert blended.shape == x.shape
    assert torch.equal(y_a, y)
    assert torch.equal(y_b, y[torch.randperm(4)]) or y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0
    assert lam >= 0.5, "the dominant label must stay readable"


def test_mixup_disabled_when_alpha_is_zero():
    x = torch.randn(4, 4)
    y = torch.arange(4)
    blended, y_a, y_b, lam = mixup_data(x, y, alpha=0.0)
    assert lam == 1.0
    assert torch.allclose(blended, x)


def test_mixup_criterion_is_the_lam_weighted_average():
    """The exact contract: lam*CE(y_a) + (1-lam)*CE(y_b)."""
    y_a = torch.tensor([0, 0, 1, 0])
    y_b = torch.tensor([1, 0, 0, 0])
    logits = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 0.0]])
    ce = nn.CrossEntropyLoss()

    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        got = mixup_criterion(logits, y_a, y_b, lam, ce)
        expected = lam * ce(logits, y_a) + (1.0 - lam) * ce(logits, y_b)
        assert torch.allclose(got, expected, atol=1e-6)


def test_mixup_criterion_falls_as_predictions_agree_with_the_target():
    y_a = torch.tensor([0, 1])
    y_b = torch.tensor([1, 0])
    ce = nn.CrossEntropyLoss()
    agreeing = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    disagreeing = torch.tensor([[0.0, 9.0], [9.0, 0.0]])
    assert mixup_criterion(agreeing, y_a, y_b, 1.0, ce) < mixup_criterion(
        disagreeing, y_a, y_b, 1.0, ce
    )


def test_mixup_criterion_actually_uses_the_supplied_criterion():
    """Regression: it once called F.cross_entropy directly, dropping class weights.

    That silently discarded the inverse-frequency weights on exactly the batches
    where the minority classes are hardest.
    """
    y_a = torch.tensor([0, 1])
    y_b = torch.tensor([1, 0])
    lam = 0.5
    logits = torch.tensor([[6.0, 0.0], [6.0, 0.0]])

    plain = mixup_criterion(logits, y_a, y_b, lam, nn.CrossEntropyLoss())
    uniform = mixup_criterion(
        logits, y_a, y_b, lam, nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    )
    weighted = mixup_criterion(
        logits, y_a, y_b, lam, nn.CrossEntropyLoss(weight=torch.tensor([5.0, 1.0]))
    )
    # Equal weights must reproduce the unweighted loss exactly...
    assert torch.allclose(plain, uniform, atol=1e-6)
    # ...and a skewed weighting must change it, proving the criterion is used.
    assert not torch.allclose(plain, weighted, atol=1e-6), (
        "class weights had no effect on the MixUp loss"
    )


def test_mixup_criterion_respects_label_smoothing():
    y = torch.tensor([0, 1])
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    plain = mixup_criterion(logits, y, y, 0.5, nn.CrossEntropyLoss())
    smooth = mixup_criterion(
        logits, y, y, 0.5, nn.CrossEntropyLoss(label_smoothing=0.2)
    )
    assert not torch.allclose(plain, smooth, atol=1e-6), (
        "label smoothing had no effect on the MixUp loss"
    )
    assert smooth > plain, "smoothing should floor the loss even when correct"


# ── class weights ───────────────────────────────────────────────────────────
def test_class_weights_favour_the_rare_class():
    # 50 majority, 2 minority
    targets = [0] * 50 + [1] * 2
    weights = compute_class_weights(targets, num_classes=2,
                                    classes=["a", "b"])
    values = [float(w) for w in (weights.values() if isinstance(weights, dict)
                                 else weights)]
    assert values[1] > values[0], "the rare class must get the larger weight"


def test_class_weights_are_finite_and_positive():
    targets = [i % 7 for i in range(70)]
    weights = compute_class_weights(targets, num_classes=7,
                                    classes=list(FINAL_CLASSES))
    values = [float(w) for w in (weights.values() if isinstance(weights, dict)
                                 else weights)]
    assert len(values) == 7
    assert all(v > 0 and v == v for v in values)


# ── end-to-end train -> infer ───────────────────────────────────────────────
def test_train_then_predict_round_trip(cfg):
    from iwnet.inference.predictor import Predictor
    from iwnet.training.trainer import run_training

    cfg.train.epochs = 1
    cfg.train.batch_size = 2
    cfg.train.mixup_alpha = 0.0
    cfg.train.label_smoothing = 0.0
    cfg.train.early_stopping_patience = 99
    cfg.model.pretrained = False

    # paths.results_dir and the checkpoints derive from dataset_root, which the
    # fixture points inside tmp_path, so nothing lands in the real project.
    result = run_training(cfg)
    assert result is not None
    assert cfg.paths.best_checkpoint.exists(), "no best checkpoint was written"
    assert cfg.paths.last_checkpoint.exists(), "no last checkpoint was written"

    predictor = Predictor(cfg, cfg.paths.best_checkpoint).load()
    images = sorted((cfg.paths.test_dir / FINAL_CLASSES[0]).iterdir())[:2]
    preds = [predictor.predict_path(p) for p in images]

    assert len(preds) == 2
    for p in preds:
        assert p.predicted_class in FINAL_CLASSES
        assert 0.0 <= p.confidence <= 1.0
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-3), (
            "probabilities must form a distribution"
        )
        assert max(p.probabilities.values()) == pytest.approx(p.confidence, abs=1e-6)
        assert set(p.probabilities) == set(FINAL_CLASSES)

    as_dict = preds[0].to_dict()
    assert as_dict["prediction"] in FINAL_CLASSES
    assert "confidence" in as_dict
    assert "not calibrated" in as_dict["confidence_note"]
