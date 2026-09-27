"""Training loop, evaluation orchestration and checkpoint management.

Bug fixes relative to the original ``run_training`` (all engineering, none
methodological):

1. **Scheduler destroyed the backbone LR multiplier.** The unfrozen backbone was
   added as ``lr=LR*0.01``, but the scheduler unconditionally wrote the *same*
   learning rate into every parameter group on its next step, so the intended
   100x-lower backbone LR never took effect. :class:`WarmupCosineScheduler` now
   stores a per-group base LR and scales each group independently.
2. **BatchNorm statistics drifted while the backbone was "frozen"** - see
   :class:`iwnet.model.architecture.IWNET`.
3. **MixUp accuracy was reported as accuracy.** It is
   ``lam*acc(pred==y_a) + (1-lam)*acc(pred==y_b)`` measured on *blended* images,
   which is neither accuracy nor comparable across epochs. It is now tracked
   separately as ``train_acc`` (mixed objective) and ``train_acc_clean`` (batches
   where MixUp was not applied), and the CSV names both explicitly.
4. **No NaN/Inf guard.** A diverged run silently poisoned the "best" checkpoint.
   Non-finite losses now stop training with existing checkpoints intact.
5. **No AMP.** A 4 GiB consumer GPU cannot sustain 224px/batch-8 EfficientNet-B3
   in fp32. Mixed precision is enabled on CUDA.
6. **Class weights used a silent ``counts.get(i, 1)`` fallback** that would emit
   a wrong weight for an absent class. An absent class is now an error.
7. **Failures destroyed work.** ``last_iwnet.pth`` is written every epoch and all
   checkpoints are written atomically, so an interrupted run never leaves a
   truncated best checkpoint and never has to start over.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from iwnet.config import FINAL_CLASSES, Config, deep_copy_config
from iwnet.model.architecture import build_model, describe_model
from iwnet.model.checkpoint import load_checkpoint, save_checkpoint
from iwnet.training import plots
from iwnet.training.metrics import (
    ClassificationMetrics,
    compute_metrics,
    write_classification_report_csv,
)
from iwnet.utils import get_device, get_logger, human_duration, set_seed

__all__ = [
    "TrainingFailure",
    "WarmupCosineScheduler",
    "EpochResult",
    "EvalResult",
    "run_training",
    "run_evaluation",
    "evaluate",
    "compute_class_weights",
    "mixup_data",
    "mixup_criterion",
]

log = get_logger("training.trainer")


class TrainingFailure(RuntimeError):
    """Raised when training cannot continue. Existing checkpoints are preserved."""


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup then cosine decay, applied *per parameter group*.

    Each group keeps its own base LR, so a group added later at a reduced rate
    (e.g. a freshly unfrozen pretrained backbone) stays reduced.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup = max(0, int(warmup_epochs))
        self.total = max(1, int(total_epochs))
        self.base_lr = float(base_lr)
        self.epoch = 0
        self.base_lrs: list[float] = [float(group["lr"]) for group in optimizer.param_groups]

    def sync(self, base_lr: float | None = None) -> None:
        """Register newly added parameter groups, leaving existing bases alone.

        Call immediately after ``optimizer.add_param_group``. Only the groups
        added since the previous call contribute new entries, using whatever LR
        they were created with.

        Re-reading *every* group's current LR here - as an earlier version did -
        captures the already-decayed warmup/cosine value as a new base, so the
        schedule gets applied to it a second time and the head's LR collapses.
        """
        if base_lr is not None:
            self.base_lr = float(base_lr)
        current = [float(group["lr"]) for group in self.optimizer.param_groups]
        if len(current) > len(self.base_lrs):
            self.base_lrs.extend(current[len(self.base_lrs) :])

    def factor(self) -> float:
        """Current schedule multiplier for the current (0-based) epoch index.

        Warmup ramps ``1/warmup .. 1`` across the first ``warmup`` epochs. Note the
        ``+1``: a plain ``epoch/warmup`` yields a factor of exactly 0 for epoch 1,
        which would train the first epoch with no learning rate at all.
        """
        if self.warmup > 0 and self.epoch < self.warmup:
            return (self.epoch + 1) / self.warmup
        # Span is total-warmup-1 so the *last* epoch lands on exactly 0.
        span = max(1, self.total - self.warmup - 1)
        progress = min(max((self.epoch - self.warmup) / span, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(self) -> float:
        """Write the scheduled LR into every group for the current epoch."""
        multiplier = self.factor()
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * multiplier
        return multiplier

    def step(self) -> float:
        self.epoch += 1
        return self.apply()


# ─────────────────────────────────────────────────────────────────────────────
#  MIXUP
# ─────────────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha: float):
    """Blend a batch with a shuffled copy of itself. Returns blended x and both label sets."""
    import numpy as np
    import torch

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)  # keep the dominant label readable
    permutation = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[permutation], y, y[permutation], lam


def mixup_criterion(logits, y_a, y_b, lam, criterion=None):
    """MixUp loss, honouring the configured criterion.

    ``criterion`` is the same module used for clean batches, so class weights and
    label smoothing also apply to mixed batches. An earlier version called
    ``F.cross_entropy`` directly, which silently discarded the inverse-frequency
    class weights on every MixUp batch - i.e. exactly the batches where the
    minority classes are hardest.
    """
    if criterion is None:
        import torch.nn.functional as F

        return lam * F.cross_entropy(logits, y_a) + (1.0 - lam) * F.cross_entropy(logits, y_b)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT TYPES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc_mixed: float
    train_acc_clean: float
    val_loss: float
    val_acc: float
    learning_rate: float
    seconds: float
    total_batches: int
    mixup_batches: int


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    metrics: ClassificationMetrics | None = None
    y_true: object = None
    y_pred: object = None
    y_probs: object = None


@dataclass
class _History:
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    train_acc_clean: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    mixup_batches: list[int] = field(default_factory=list)
    total_batches: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "train_acc_clean": self.train_acc_clean,
            "val_acc": self.val_acc,
            "lr": self.lr,
            "epoch_seconds": self.epoch_seconds,
            "mixup_batches": self.mixup_batches,
            "total_batches": self.total_batches,
        }

    def extend(self, result: EpochResult) -> None:
        self.epoch.append(result.epoch)
        self.train_loss.append(result.train_loss)
        self.val_loss.append(result.val_loss)
        self.train_acc.append(result.train_acc_mixed)
        self.train_acc_clean.append(result.train_acc_clean)
        self.val_acc.append(result.val_acc)
        self.lr.append(result.learning_rate)
        self.epoch_seconds.append(result.seconds)
        self.mixup_batches.append(result.mixup_batches)
        self.total_batches.append(result.total_batches)


# ─────────────────────────────────────────────────────────────────────────────
#  AMP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _amp_enabled(device, requested: bool) -> bool:
    return bool(requested) and device.type == "cuda"


def _autocast(device, enabled: bool):
    import torch

    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _make_scaler(device, requested: bool):
    import torch

    return torch.amp.GradScaler(device.type, enabled=_amp_enabled(device, requested))


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, cfg: Config, scaler):
    """One training epoch.

    Returns ``(loss, mixed_acc, clean_acc, mixup_batches, total_batches)``.
    """
    import torch

    model.train()
    total_loss = 0.0
    total_samples = 0
    mixed_correct = 0.0
    mixed_samples = 0
    clean_correct = 0
    clean_samples = 0
    mixup_batches = 0
    total_batches = 0
    use_amp = scaler.is_enabled()
    accum = max(1, int(cfg.train.grad_accum_steps))
    backbone_frozen_at_start = model._backbone_frozen

    optimizer.zero_grad(set_to_none=True)

    for batch_index, (images, labels) in enumerate(loader):
        total_batches += 1
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        apply_mixup = random.random() < cfg.train.mixup_probability and batch_size > 1
        if apply_mixup:
            mixup_batches += 1
            blended, y_a, y_b, lam = mixup_data(images, labels, cfg.train.mixup_alpha)
            with _autocast(device, use_amp):
                logits = model(blended)
                loss = mixup_criterion(logits, y_a, y_b, lam, criterion)
            # Soft blended objective - reported separately and labelled as such.
            mixed_correct += (
                lam * (logits.argmax(1) == y_a).sum().item()
                + (1.0 - lam) * (logits.argmax(1) == y_b).sum().item()
            )
            mixed_samples += batch_size
        else:
            with _autocast(device, use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            clean_correct += (logits.argmax(1) == labels).sum().item()
            clean_samples += batch_size

        if not torch.isfinite(loss):
            raise TrainingFailure(
                f"Non-finite loss ({loss.item():.6g}) at batch {batch_index + 1}/{len(loader)}.\n"
                "  Training stopped. Existing checkpoints, logs and the dataset are intact.\n"
                "  Likely causes: learning rate too high, a corrupt image, or a numerical\n"
                "  instability. Try lowering train.learning_rate, or pass --no-amp to rule\n"
                "  out mixed-precision overflow."
            )

        scaler.scale(loss / accum).backward()

        is_last = (batch_index + 1) == len(loader)
        if (batch_index + 1) % accum == 0 or is_last:
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        interval = cfg.train.log_every_n_batches
        if interval and (batch_index + 1) % interval == 0:
            print(
                f"      batch {batch_index + 1:>5}/{len(loader)}  "
                f"loss={total_loss / max(total_samples, 1):.4f}",
                end="\r",
                flush=True,
            )
    print(" " * 70, end="\r")

    if total_samples == 0:
        raise TrainingFailure("Training loader yielded no samples.")

    # mixed_acc is the blended-soft-label objective score, so its denominator must
    # be the number of MIXED samples. Dividing by total_samples (which also
    # counts clean batches) understated it by the clean-batch fraction.
    mixed_acc = mixed_correct / mixed_samples if mixed_samples else float("nan")
    clean_acc = clean_correct / clean_samples if clean_samples else float("nan")
    return total_loss / total_samples, mixed_acc, clean_acc, mixup_batches, total_batches


def evaluate(model, loader, criterion, device, cfg: Config, classes: Sequence[str]) -> EvalResult:
    """Deterministic evaluation pass (no augmentation, fp32, ``inference_mode``)."""
    import numpy as np
    import torch

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_true, all_pred, all_probs = [], [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits.float(), dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())

    if total == 0:
        log.warning("Evaluation loader yielded no samples")
        return EvalResult(loss=float("nan"), accuracy=float("nan"))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_probs = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_probs, list(classes))

    return EvalResult(
        loss=total_loss / total,
        accuracy=correct / total,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(targets: Sequence[int], num_classes: int, classes: Sequence[str]):
    """Inverse-frequency weights. An absent class is an error, not a silent 1.0."""
    import torch

    counts = [0] * num_classes
    for label in targets:
        counts[int(label)] += 1

    absent = [classes[i] for i, count in enumerate(counts) if count == 0]
    if absent:
        raise TrainingFailure(
            f"Cannot compute class weights - no training samples for: {absent}.\n"
            "All 7 classes must be present in the training split."
        )

    total = sum(counts)
    return torch.tensor([total / (num_classes * c) for c in counts], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FORMATTING
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_pct(value: float) -> str:
    if value != value:  # NaN - e.g. no clean batches this epoch
        return "N/A"
    return f"{value * 100:.2f}%"


def _free_cuda(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()


def _read_fingerprint(cfg: Config) -> str | None:
    """Dataset fingerprint recorded at build time, tying a checkpoint to its data."""
    import csv

    path = cfg.paths.results_dir / "dataset_report.csv"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("metric") == "dataset_fingerprint_sha256":
                    return row.get("value")
    except OSError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  FULL TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────
def run_training(cfg: Config, *, resume: bool = False) -> dict:
    """Train, save checkpoints, then evaluate the best model on the test set."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from iwnet.data.dataset import build_loaders
    from iwnet.data.validation import validate_dataset

    print("=" * 74)
    print("  TRAINING")
    print("=" * 74)

    # Validate the dataset before anything expensive happens.
    report = validate_dataset(cfg, deep=cfg.data.fail_on_corrupt)
    if not report.ok:
        for problem in report.fatal:
            print(f"  [ERROR] {problem}")
        raise TrainingFailure("Dataset validation failed. Nothing was trained.")
    print(f"  Dataset verified: {report.train_total} train / {report.test_total} test images, "
          f"{len(report.train_counts)} classes\n")

    device = get_device()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device : {device} ({device_name})")
    print(f"  AMP    : {'enabled' if _amp_enabled(device, cfg.train.amp) else 'disabled'}\n")

    train_loader, val_loader, test_loader, classes, split_info = build_loaders(cfg)
    for cls, reason in split_info.starved_classes.items():
        print(f"  [WARN] Class '{cls}' contributes no validation samples ({reason}).")

    model = build_model(cfg).to(device)
    print()
    print(describe_model(model))
    print()

    train_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    if cfg.train.use_class_weights:
        weights = compute_class_weights(
            train_loader.dataset.targets, len(classes), classes
        ).to(device)
        train_criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.train.label_smoothing, weight=weights
        )
        print(f"  Class weights: {[round(float(w), 3) for w in weights]}")
    # Plain CE for reported metrics: no label smoothing or re-weighting, so the
    # loss number is directly interpretable.
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.train.warmup_epochs, cfg.train.epochs, cfg.train.learning_rate
    )
    scaler = _make_scaler(device, cfg.train.amp)

    start_epoch = 1
    best_val_acc = 0.0
    history = _History()
    fingerprint = _read_fingerprint(cfg)

    if resume:
        start_epoch, best_val_acc = _resume_from(cfg, model, optimizer, scheduler, device, classes)
        # The resumed model may already have an unfrozen backbone.
        if start_epoch > cfg.train.unfreeze_epoch:
            model._backbone_frozen = False

    print(f"\n  Epochs {start_epoch}..{cfg.train.epochs} | batch={cfg.train.batch_size} | "
          f"image={cfg.train.image_size} | seed={cfg.data.seed}")
    print(f"  Backbone unfreezes at epoch {cfg.train.unfreeze_epoch} "
          f"(last {cfg.train.unfreeze_stages} stages, LR x{cfg.train.backbone_lr_scale})")
    print(f"  Dataset fingerprint: {fingerprint or 'not recorded (rebuild the dataset)'}")
    print("=" * 74)
    print(f"  {'EPOCH':>6}  {'TRAINLOSS':>10}  {'TRAINACC*':>10}  {'CLEANTRAIN':>10}  "
          f"{'VALLOSS':>9}  {'VALACC':>8}  {'LR':>10}  {'TIME':>7}")
    print("  * MixUp soft objective, not classification accuracy.")
    print()

    no_improve = 0
    pending_unfreeze = bool(cfg.model.freeze_backbone) and bool(cfg.train.unfreeze_epoch)

    # On resume, fast-forward the schedule so the LR matches the epoch we are on.
    scheduler.epoch = max(0, start_epoch - 1)
    # Write the warmup LR into the optimizer *before* epoch 1. Without this the
    # first epoch runs at the full base rate, so warmup only ever affected
    # epochs 2..warmup and the documented ramp never actually happened.
    scheduler.apply()

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]

        if pending_unfreeze and epoch == cfg.train.unfreeze_epoch:
            unfrozen = model.unfreeze_backbone(cfg.train.unfreeze_stages)
            pending_unfreeze = False
            if unfrozen != 0:
                new_params = [p for p in model.backbone.parameters() if p.requires_grad]
                scaled_lr = cfg.train.learning_rate * cfg.train.backbone_lr_scale
                optimizer.add_param_group({"params": new_params, "lr": scaled_lr})
                # Register the new group's base without disturbing the head's.
                scheduler.sync()
                scheduler.apply()
                print(f"  Backbone unfrozen. LR x{cfg.train.backbone_lr_scale} = {scaled_lr:.3e}")
                print()

        try:
            train_loss, mixed_acc, clean_acc, mixup_batches, total_batches = train_one_epoch(
                model, train_loader, train_criterion, optimizer, device, cfg, scaler
            )
        except torch.cuda.OutOfMemoryError as exc:
            _free_cuda(device)
            raise TrainingFailure(
                f"CUDA out of memory during epoch {epoch}: {exc}\n"
                "  Checkpoints, logs and the dataset are intact.\n"
                "  Reduce train.batch_size (try 4), lower train.image_size, or pass --no-amp\n"
                "  if mixed precision was not already enabled."
            ) from exc

        val_result = evaluate(model, val_loader, train_criterion, device, cfg, classes)
        scheduler.step()
        elapsed = time.perf_counter() - started

        if not math.isfinite(val_result.loss) or not math.isfinite(val_result.accuracy):
            raise TrainingFailure(
                f"Non-finite validation metrics at epoch {epoch} "
                f"(loss={val_result.loss}, acc={val_result.accuracy}). Checkpoints are intact."
            )

        history.extend(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_acc_mixed=mixed_acc,
                train_acc_clean=clean_acc,
                val_loss=val_result.loss,
                val_acc=val_result.accuracy,
                learning_rate=learning_rate,
                seconds=elapsed,
                total_batches=total_batches,
                mixup_batches=mixup_batches,
            )
        )

        improved = val_result.accuracy > best_val_acc
        print(
            f"  {epoch:>6}  {train_loss:>10.4f}  {mixed_acc * 100:>9.2f}%  "
            f"{_fmt_pct(clean_acc):>10}  {val_result.loss:>9.4f}  "
            f"{val_result.accuracy * 100:>7.2f}%  {learning_rate:>10.2e}  "
            f"{human_duration(elapsed):>7}{'  <- BEST' if improved else ''}"
        )

        if improved:
            best_val_acc = val_result.accuracy
            save_checkpoint(
                cfg.paths.best_checkpoint, model.state_dict(), classes, cfg,
                epoch=epoch, val_acc=val_result.accuracy, dataset_fingerprint=fingerprint,
                device=str(device), train_acc=clean_acc, val_loss=val_result.loss,
                extra={"mixup_batches": mixup_batches, "total_batches": total_batches},
            )
            no_improve = 0
        else:
            no_improve += 1

        # `last` every epoch so any interruption is resumable.
        save_checkpoint(
            cfg.paths.last_checkpoint, model.state_dict(), classes, cfg,
            epoch=epoch, val_acc=val_result.accuracy, dataset_fingerprint=fingerprint,
            device=str(device), train_acc=clean_acc, val_loss=val_result.loss,
            optimizer_state=optimizer.state_dict(),
            scheduler_state={"epoch": scheduler.epoch, "base_lrs": scheduler.base_lrs},
        )

        if no_improve >= cfg.train.early_stop_patience:
            print(f"\n  Early stopping: no validation improvement for {no_improve} epochs.")
            break

    print("\n" + "=" * 74)
    print(f"  Training finished. Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Best checkpoint: {cfg.paths.best_checkpoint}")
    print(f"  Last checkpoint: {cfg.paths.last_checkpoint}")
    print("=" * 74)

    history_dict = history.as_dict()
    plots.write_training_history_csv(history_dict, cfg)
    plots.plot_loss_curve(history_dict, cfg)
    plots.plot_accuracy_curve(history_dict, cfg)
    plots.plot_lr_schedule(history_dict, cfg)
    plots.plot_dataset_distribution(cfg, classes, split_info)

    # Evaluate the BEST weights on the untouched test split.
    payload = load_checkpoint(
        cfg.paths.best_checkpoint, device=str(device), expected_classes=classes, cfg=cfg
    )
    model.load_state_dict(payload["model_state"])
    test_result = run_evaluation(cfg, model, test_loader, classes, device, "best checkpoint")

    return {
        "best_val_acc": best_val_acc,
        "epochs_run": len(history.epoch),
        "history": history_dict,
        "test_metrics": test_result.metrics if test_result else None,
        "dataset_fingerprint": fingerprint,
        "best_checkpoint": str(cfg.paths.best_checkpoint),
    }


def _resume_from(cfg, model, optimizer, scheduler, device, classes) -> tuple[int, float]:
    """Restore model/optimizer/scheduler state from ``last_iwnet.pth``."""
    if not cfg.paths.last_checkpoint.is_file():
        print(f"  [INFO] Nothing to resume from ({cfg.paths.last_checkpoint} not found).")
        return 1, 0.0

    payload = load_checkpoint(
        cfg.paths.last_checkpoint, device=str(device), expected_classes=classes, cfg=cfg
    )
    model.load_state_dict(payload["model_state"])

    if "optimizer_state" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except ValueError as exc:
            log.warning("Optimizer state incompatible (%s); starting with a fresh optimizer.", exc)
    if "scheduler_state" in payload:
        state = payload["scheduler_state"]
        scheduler.epoch = int(state.get("epoch", 0))
        stored_bases = state.get("base_lrs")
        if stored_bases and len(stored_bases) == len(optimizer.param_groups):
            scheduler.base_lrs = list(stored_bases)

    start = int(payload.get("epoch", 0)) + 1
    previous = float(payload.get("val_acc", 0.0))
    print(f"  Resuming after epoch {payload.get('epoch')} (next epoch {start}); "
          f"previous val acc {previous * 100:.2f}%")
    if start > cfg.train.epochs:
        print(f"  [WARN] Epoch budget ({cfg.train.epochs}) already reached.")
    return start, previous


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION ONLY
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    cfg: Config,
    model,
    test_loader,
    classes: Sequence[str],
    device,
    checkpoint_label: str = "checkpoint",
) -> EvalResult | None:
    """Evaluate a model on the test split and write every report and figure."""
    import torch.nn as nn

    print(f"\n  Evaluating on the TEST split ({checkpoint_label})...")
    print("  The test split is read-only: no augmentation, no training, no tuning.")

    result = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, cfg, classes)
    print(f"  Test loss {result.loss:.4f}   accuracy {result.accuracy * 100:.2f}%")

    if result.metrics is None:
        print("  [ERROR] Could not compute metrics (empty test set).")
        return result

    result.metrics.print_summary("TEST")
    write_classification_report_csv(
        result.metrics, cfg.paths.results_dir / "classification_report.csv"
    )

    plots.plot_confusion_matrix(result.metrics, cfg)
    plots.plot_confusion_matrix_normalized(result.metrics, cfg)
    plots.plot_per_class_accuracy(result.metrics, cfg)
    plots.plot_precision_recall_f1(result.metrics, cfg)
    plots.plot_roc_auc(result.metrics, result.y_true, result.y_probs, cfg)
    plots.plot_sample_predictions(model, test_loader, classes, device, cfg)
    plots.plot_class_confidence(result.metrics, result.y_true, result.y_probs, cfg)

    print(f"\n  Figures and reports -> {cfg.paths.results_dir}")
    return result


def evaluate_checkpoint(cfg: Config, path: Path | None = None) -> EvalResult | None:
    """Evaluate a saved checkpoint on the test split. No training, no tuning.

    This is the ``--evaluate`` entry point: it rebuilds the model from the
    checkpoint's own stored geometry (so a checkpoint always matches its own
    architecture), builds the test loader, and writes every report and figure.
    """
    import torch
    import torch.nn as nn

    from iwnet.data.dataset import build_loaders
    from iwnet.model.architecture import build_model

    checkpoint = Path(path) if path is not None else cfg.paths.best_checkpoint
    if not checkpoint.is_file():
        print(f"\n[ERROR] No checkpoint at {checkpoint}")
        print("  Train one first:  python grape.py\n")
        return None

    set_seed(cfg.data.seed)
    device = get_device()

    print("=" * 74)
    print(f"  EVALUATING CHECKPOINT: {checkpoint.name}")
    print("=" * 74)

    payload = load_checkpoint(checkpoint, device=str(device), expected_classes=FINAL_CLASSES)
    classes = list(payload["classes"])
    num_classes = len(classes)

    # Rebuild the architecture AND the preprocessing from the checkpoint's own
    # stored config. Using the *current* cfg here means a checkpoint trained at
    # a different image size or with a different backbone either fails with an
    # opaque state_dict shape error or, worse, is evaluated under the wrong
    # transform and reports numbers that were never measured.
    from iwnet.inference.predictor import Predictor

    eval_cfg = Predictor._config_from_checkpoint(payload)
    eval_cfg.data.seed = cfg.data.seed
    image_size = int(payload.get("image_size", eval_cfg.train.image_size))
    eval_cfg.train.image_size = image_size

    backbone = payload.get("config", {}).get("model", {}).get(
        "backbone", cfg.model.backbone
    )
    if backbone != eval_cfg.model.backbone:
        print(f"\n[NOTE] Checkpoint uses backbone '{backbone}', not the current "
              f"'{eval_cfg.model.backbone}'.")
        eval_cfg.model.backbone = backbone

    model = build_model(eval_cfg, pretrained=False, num_classes=num_classes)
    model.load_state_dict(payload["model_state"])
    model.eval()
    model.to(device)

    _train_loader, _val_loader, test_loader, _classes, _split_info = build_loaders(eval_cfg)
    print(f"  Architecture: IWNET / {backbone} | image_size={image_size}")
    print(f"  Epoch saved : {payload.get('epoch')} | val acc {float(payload.get('val_acc', 0.0)) * 100:.2f}%")
    print(f"  Classes     : {num_classes}")
    fingerprint = _read_fingerprint(cfg)
    if fingerprint:
        print(f"  Dataset     : fingerprint {fingerprint}")
    print("=" * 74)

    return run_evaluation(eval_cfg, model, test_loader, classes, device, checkpoint.name)
