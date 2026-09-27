"""Matplotlib figures for training history and test-set evaluation.

All figures are written to ``results/`` and the backend is forced to ``Agg`` so
this works headless. Every plot tolerates degenerate input (a class with zero
test samples, a single-epoch history) without crashing - a missing metric is
drawn as an explicit "N/A" marker, never as a fabricated bar.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from iwnet.config import Config
from iwnet.training.metrics import NA, ClassificationMetrics

__all__ = [
    "MatplotlibSetup",
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_lr_schedule",
    "plot_dataset_distribution",
    "plot_confusion_matrix",
    "plot_confusion_matrix_normalized",
    "plot_per_class_accuracy",
    "plot_precision_recall_f1",
    "plot_roc_auc",
    "plot_sample_predictions",
    "plot_class_confidence",
    "write_training_history_csv",
    "PLOT_FILES",
]

PLOT_FILES = (
    "loss_curve.png",
    "accuracy_curve.png",
    "lr_schedule.png",
    "dataset_distribution.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "per_class_accuracy.png",
    "precision_recall_f1.png",
    "roc_curves.png",
    "sample_predictions.png",
    "class_confidence.png",
)


class MatplotlibSetup:
    """Import matplotlib with a headless backend, once."""

    _ready = False

    @classmethod
    def get(cls):
        if not cls._ready:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cls._ready = True
            return plt
        import matplotlib.pyplot as plt

        return plt


def _save(plt, results_dir: Path, name: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def _has_history(history: dict, key: str) -> bool:
    return bool(history.get(key))


def plot_loss_curve(history: dict, cfg: Config):
    plt = MatplotlibSetup.get()
    epochs = history.get("epoch", [])
    if not epochs:
        return None
    plt.figure(figsize=(8, 5))
    for key, label, colour in (
        ("train_loss", "Train loss", "#5B8DB8"),
        ("val_loss", "Validation loss", "#E07B54"),
    ):
        if _has_history(history, key):
            plt.plot(epochs, history[key], marker="o", label=label, color=colour)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "loss_curve.png")


def plot_accuracy_curve(history: dict, cfg: Config):
    plt = MatplotlibSetup.get()
    epochs = history.get("epoch", [])
    if not epochs:
        return None
    plt.figure(figsize=(8, 5))
    for key, label, colour in (
        ("train_acc", "Train (mixed objective)", "#5B8DB8"),
        ("train_acc_clean", "Train (clean batches only)", "#8FB8DE"),
        ("val_acc", "Validation", "#4C9F70"),
    ):
        if _has_history(history, key):
            style = "--" if key == "train_acc_clean" else "-"
            marker = "s" if key == "train_acc_clean" else "o"
            plt.plot(epochs, [v * 100 for v in history[key]], marker=marker, linestyle=style,
                     label=label, color=colour)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "accuracy_curve.png")


def plot_lr_schedule(history: dict, cfg: Config):
    plt = MatplotlibSetup.get()
    epochs = history.get("epoch", [])
    lrs = history.get("lr", [])
    if not epochs or not lrs or max(lrs) <= 0:
        return None
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lrs, marker="o", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.yscale("log")
    plt.title("Learning-Rate Schedule (warmup + cosine)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "lr_schedule.png")


def write_training_history_csv(history: dict, cfg: Config) -> Path:
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.results_dir / "training_history.csv"
    keys = [
        "epoch", "train_loss", "val_loss", "train_acc", "train_acc_clean",
        "val_acc", "lr", "epoch_seconds", "mixup_batches", "total_batches",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for index in range(len(history.get("epoch", []))):
            writer.writerow(
                [
                    _cell(history, key, index)
                    for key in keys
                ]
            )
    return path


def _cell(history: dict, key: str, index: int) -> str:
    values = history.get(key) or []
    if index >= len(values):
        return ""
    value = values[index]
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
def plot_dataset_distribution(cfg: Config, classes: Sequence[str], split_info=None):
    """Train/validation/test per class.

    Counts images by extension and recurses, so it agrees with the verified
    counts. The original used a bare ``os.listdir`` which counted non-image
    files and could disagree with ``_count_images_recursive``.
    """
    from iwnet.data.validation import count_split

    plt = MatplotlibSetup.get()
    train = count_split(cfg.paths.train_dir, cfg.data.img_exts)
    test = count_split(cfg.paths.test_dir, cfg.data.img_exts)
    validation = split_info.val_counts if split_info is not None else {}

    import numpy as np

    x = np.arange(len(classes))
    width = 0.27
    train_values = [train.get(c, 0) for c in classes]
    test_values = [test.get(c, 0) for c in classes]
    val_values = [validation.get(c, 0) for c in classes]

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, train_values, width, label="Train", color="#5B8DB8")
    if any(val_values):
        plt.bar(x, val_values, width, label="Validation (from train)", color="#AED6F1")
    plt.bar(x + width, test_values, width, label="Test", color="#4C9F70")
    plt.xticks(x, classes, rotation=25, ha="right")
    plt.ylabel("Images")
    plt.title("Dataset Distribution")
    plt.legend()
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "dataset_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(metrics: ClassificationMetrics, cfg: Config):
    from sklearn.metrics import ConfusionMatrixDisplay

    plt = MatplotlibSetup.get()
    matrix = metrics.confusion_matrix
    if not matrix:
        return None
    # ConfusionMatrixDisplay calls .shape, so it needs an ndarray, not a list.
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(
        np.asarray(matrix, dtype=np.int64), display_labels=metrics.classes
    ).plot(
        ax=ax, cmap="Blues", colorbar=True, xticks_rotation=30
    )
    ax.set_title("Confusion Matrix (test set, counts)")
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "confusion_matrix.png")


def plot_confusion_matrix_normalized(metrics: ClassificationMetrics, cfg: Config):
    """Row-normalised confusion matrix - shows per-class behaviour, not totals."""
    from sklearn.metrics import ConfusionMatrixDisplay

    plt = MatplotlibSetup.get()
    matrix = metrics.confusion_matrix
    if not matrix:
        return None
    import numpy as np

    array = np.asarray(matrix, dtype=float)
    totals = array.sum(axis=1, keepdims=True)
    # Rows with no samples stay all-zero rather than becoming NaN.
    normalised = np.divide(array, totals, out=np.zeros_like(array), where=totals != 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(normalised, display_labels=metrics.classes).plot(
        ax=ax, cmap="Greens", colorbar=True, xticks_rotation=30, values_format=".2f"
    )
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "confusion_matrix_normalized.png")


def plot_per_class_accuracy(metrics: ClassificationMetrics, cfg: Config):
    plt = MatplotlibSetup.get()
    if not metrics.per_class:
        return None
    names = [row.name for row in metrics.per_class]
    values = [row.accuracy for row in metrics.per_class]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(len(names))
    defined = [(i, v) for i, v in enumerate(values) if v is not None]
    if defined:
        bars = ax.bar([i for i, _ in defined], [v * 100 for _, v in defined], color="#4C9F70")
        for bar, (_, value) in zip(bars, defined):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1.5,
                    f"{value * 100:.1f}%", ha="center", fontsize=9)
    for index, value in enumerate(values):
        if value is None:
            ax.text(index, 3, NA, ha="center", fontsize=11, color="#B03A2E", fontweight="bold")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Recall / per-class accuracy (%)")
    ax.set_title("Per-Class Accuracy (undefined classes marked N/A)")
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "per_class_accuracy.png")


def plot_precision_recall_f1(metrics: ClassificationMetrics, cfg: Config):
    plt = MatplotlibSetup.get()
    if not metrics.per_class:
        return None
    import numpy as np

    names = [row.name for row in metrics.per_class]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, key, label, colour in (
        (-width, "precision", "Precision", "#5B8DB8"),
        (0.0, "recall", "Recall", "#4C9F70"),
        (width, "f1", "F1-score", "#E07B54"),
    ):
        values = [getattr(row, key) for row in metrics.per_class]
        drawable = [(i, v) for i, v in enumerate(values) if v is not None]
        if drawable:
            ax.bar([i + offset for i, _ in drawable], [v for _, v in drawable],
                   width, label=label, color=colour)
        for index, value in enumerate(values):
            if value is None:
                ax.text(index + offset, 0.02, "N/A", ha="center", fontsize=7,
                        rotation=90, color="#B03A2E")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Class")
    ax.legend()
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "precision_recall_f1.png")


def plot_roc_auc(metrics: ClassificationMetrics, y_true, y_probs, cfg: Config):
    plt = MatplotlibSetup.get()
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    if y_probs is None:
        return None
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    if y_probs.ndim != 2 or y_probs.shape[0] != y_true.size:
        return None

    n_classes = len(metrics.classes)
    binary = np.zeros((y_true.size, n_classes), dtype=int)
    binary[np.arange(y_true.size), y_true] = 1

    fig, ax = plt.subplots(figsize=(8, 7))
    drawn = 0
    for i, name in enumerate(metrics.classes):
        if binary[:, i].sum() == 0 or binary[:, i].sum() == y_true.size:
            continue
        fpr, tpr, _ = roc_curve(binary[:, i], y_probs[:, i])
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={score:.3f})")
        drawn += 1
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if drawn:
        ax.set_title("ROC Curves (one-vs-rest)")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.set_title("ROC Curves - undefined: every class is all-positive or all-negative")
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "roc_curves.png")


def plot_class_confidence(metrics: ClassificationMetrics, y_true, y_probs, cfg: Config):
    """Box plot of predicted confidence per ground-truth class."""
    plt = MatplotlibSetup.get()
    import numpy as np

    if y_probs is None:
        return None
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    if y_probs.ndim != 2 or y_probs.shape[0] != y_true.size:
        return None

    data = [y_probs[y_true == i, i] for i in range(len(metrics.classes))]
    labels = [name if len(values) else f"{name} ({NA})" for name, values in zip(metrics.classes, data)]
    if not any(len(values) for values in data):
        return None

    fig, ax = plt.subplots(figsize=(11, 5))
    # Matplotlib >= 3.9 renamed boxplot's `labels` argument to `tick_labels`.
    try:
        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1"))
    except TypeError:
        ax.boxplot(data, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1"))
    ax.set_ylabel("Model confidence (softmax)")
    ax.set_title("Prediction Confidence per Ground-Truth Class")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "class_confidence.png")


def plot_sample_predictions(model, loader, classes, device, cfg: Config, limit: int = 10):
    """A grid of correctly and incorrectly classified test images."""
    import numpy as np
    import torch

    plt = MatplotlibSetup.get()
    from iwnet.data.dataset import IMAGENET_MEAN, IMAGENET_STD

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    model.eval()
    tiles: list[tuple] = []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            confidences, predictions = probabilities.max(1)
            for i in range(images.size(0)):
                if len(tiles) >= limit:
                    break
                image = np.clip(std * images[i].permute(1, 2, 0).numpy() + mean, 0, 1)
                actual = classes[int(labels[i])]
                predicted = classes[int(predictions[i])]
                tiles.append(
                    (image, actual, predicted, float(confidences[i]), actual == predicted)
                )
            if len(tiles) >= limit:
                break

    if not tiles:
        return None

    columns = 5
    rows = (len(tiles) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.6 * rows))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, (image, actual, predicted, confidence, correct) in zip(axes, tiles):
        ax.imshow(image)
        ax.axis("off")
        marker = "OK" if correct else "MISS"
        ax.set_title(
            f"[{marker}] actual: {actual}\npred: {predicted}\nconf: {confidence * 100:.1f}%",
            fontsize=8,
            color="#1E7B45" if correct else "#B03A2E",
        )
    for ax in axes[len(tiles):]:
        ax.axis("off")
    plt.suptitle("Sample Test-Set Predictions (softmax confidence, not calibrated)", y=1.0)
    plt.tight_layout()
    return _save(plt, cfg.paths.results_dir, "sample_predictions.png")
