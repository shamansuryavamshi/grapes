"""Evaluation metrics with honest edge-case handling.

Every metric here can legitimately be *undefined* - a class absent from the test
set, an empty prediction set, a single-sample class. In those cases this module
returns ``None`` and the reports render ``N/A``. It never substitutes 0.0, 1.0,
0.5 or any other plausible-looking number, because a fabricated metric is worse
than a missing one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = [
    "PerClassMetric",
    "ClassificationMetrics",
    "safe_div",
    "compute_metrics",
    "format_value",
    "write_classification_report_csv",
]

NA = "N/A"


def safe_div(numerator: float, denominator: float) -> float | None:
    """Division that returns ``None`` instead of raising or inventing a value."""
    if denominator == 0:
        return None
    return numerator / denominator


def format_value(value: float | None, *, precision: int = 4, as_percent: bool = False) -> str:
    if value is None:
        return NA
    if as_percent:
        return f"{value * 100:.{precision - 1}f}%"
    return f"{value:.{precision}f}"


@dataclass
class PerClassMetric:
    name: str
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    support: int = 0
    auc: float | None = None
    accuracy: float | None = None
    #: True when the class had no ground-truth samples and metrics are undefined.
    undefined: bool = False

    def as_row(self) -> list[str]:
        return [
            self.name,
            format_value(self.precision),
            format_value(self.recall),
            format_value(self.f1),
            format_value(self.auc),
            format_value(self.accuracy, as_percent=True),
            str(self.support),
        ]


@dataclass
class ClassificationMetrics:
    """Complete evaluation of one split."""

    classes: list[str] = field(default_factory=list)
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    per_class: list[PerClassMetric] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    total_samples: int = 0
    notes: list[str] = field(default_factory=list)

    def class_row(self, name: str) -> PerClassMetric | None:
        for row in self.per_class:
            if row.name == name:
                return row
        return None

    def print_summary(self, title: str = "TEST") -> None:
        # ASCII rules only. A non-ASCII separator (U+2500 etc.) raises
        # UnicodeEncodeError on a Windows cp1252 console, which previously
        # crashed the run *after* training had finished and the checkpoint was
        # already saved - losing the whole evaluation report.
        bar = "-" * 74
        print(f"\n{title} SET METRICS  ({self.total_samples} samples)")
        print(bar)
        print(f"  {'CLASS':<17}{'PREC':>9}{'RECALL':>9}{'F1':>9}{'AUC':>9}{'ACC':>9}{'N':>7}")
        for row in self.per_class:
            print(
                f"  {row.name:<17}"
                f"{format_value(row.precision):>9}"
                f"{format_value(row.recall):>9}"
                f"{format_value(row.f1):>9}"
                f"{format_value(row.auc):>9}"
                f"{format_value(row.accuracy, as_percent=True):>9}"
                f"{row.support:>7}"
            )
        print(bar)
        print(f"  {'OVERALL':<17}")
        print(f"    Accuracy          : {format_value(self.accuracy, as_percent=True)}")
        print(f"    Balanced accuracy : {format_value(self.balanced_accuracy, as_percent=True)}")
        print(f"    Macro F1          : {format_value(self.macro_f1)}")
        print(f"    Weighted F1       : {format_value(self.weighted_f1)}")
        print(f"    Macro precision   : {format_value(self.macro_precision)}")
        print(f"    Macro recall      : {format_value(self.macro_recall)}")
        for note in self.notes:
            print(f"    [note] {note}")
        print(bar)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_probs,
    classes: Sequence[str],
) -> ClassificationMetrics:
    """Full metric set for one split.

    ``y_probs`` may be ``None`` if probabilities were not collected; AUC fields
    then stay ``None`` and render as ``N/A``.
    """
    import numpy as np
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(classes)
    metrics = ClassificationMetrics(classes=list(classes), total_samples=int(y_true.size))

    if y_true.size == 0:
        metrics.notes.append("Empty evaluation set - every metric is undefined.")
        return metrics

    labels = list(range(n_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics.confusion_matrix = matrix.tolist()

    metrics.accuracy = safe_div(int((y_true == y_pred).sum()), int(y_true.size))

    # Present-class recall average. A class with no ground-truth sample is
    # excluded rather than counted as 0.
    recalls_present = [
        safe_div(matrix[i, i], matrix[i].sum()) for i in labels if matrix[i].sum() > 0
    ]
    recalls_present = [r for r in recalls_present if r is not None]
    metrics.balanced_accuracy = (
        float(np.mean(recalls_present)) if recalls_present else None
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # ROC-AUC (one-vs-rest), skipped where undefined.
    aucs: list[float | None] = [None] * n_classes
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float)
        if y_probs.ndim == 2 and y_probs.shape[1] == n_classes and y_probs.shape[0] == y_true.size:
            from sklearn.metrics import auc, roc_curve

            binary = np.zeros((y_true.size, n_classes), dtype=int)
            binary[np.arange(y_true.size), y_true] = 1
            for i in range(n_classes):
                positives = int(binary[:, i].sum())
                negatives = int(y_true.size - positives)
                if positives == 0 or negatives == 0:
                    continue  # undefined -> N/A
                try:
                    fpr, tpr, _ = roc_curve(binary[:, i], y_probs[:, i])
                    aucs[i] = float(auc(fpr, tpr))
                except ValueError:
                    continue
        else:
            metrics.notes.append("Probability matrix shape mismatch - AUC reported as N/A.")

    missing_truth = [classes[i] for i in labels if matrix[i].sum() == 0]
    if missing_truth:
        metrics.notes.append(
            f"No ground-truth samples for: {', '.join(missing_truth)} - their metrics are N/A."
        )

    for i, name in enumerate(classes):
        row = PerClassMetric(name=name, support=int(support[i]), auc=aucs[i])
        if matrix[i].sum() == 0:
            row.undefined = True
        else:
            # precision/f1 can legitimately be 0; recall is 0/0 only when the
            # class is absent, which is already handled.
            row.precision = float(precision[i]) if support[i] > 0 or matrix[:, i].sum() > 0 else None
            row.recall = safe_div(matrix[i, i], matrix[i].sum())
            row.f1 = float(f1[i])
            row.accuracy = row.recall  # per-class one-vs-rest accuracy == recall
        metrics.per_class.append(row)

    defined = [r.f1 for r in metrics.per_class if r.f1 is not None]
    metrics.macro_f1 = float(np.mean(defined)) if defined else None

    defined_prec = [r.precision for r in metrics.per_class if r.precision is not None]
    defined_rec = [r.recall for r in metrics.per_class if r.recall is not None]
    metrics.macro_precision = float(np.mean(defined_prec)) if defined_prec else None
    metrics.macro_recall = float(np.mean(defined_rec)) if defined_rec else None

    total_support = int(support.sum())
    if total_support > 0:
        metrics.weighted_f1 = float(np.average(f1, weights=support))
    else:
        metrics.weighted_f1 = None

    if len(missing_truth) == n_classes:
        metrics.notes.append("No class had any ground-truth sample - class metrics are meaningless.")

    return metrics


def write_classification_report_csv(metrics: ClassificationMetrics, path: Path) -> Path:
    """Per-class metrics as CSV. Writes ``N/A`` rather than a number when undefined."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["class", "precision", "recall", "f1", "auc", "per_class_accuracy", "support"]
        )
        for row in metrics.per_class:
            writer.writerow(row.as_row())
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", format_value(metrics.accuracy)])
        writer.writerow(["balanced_accuracy", format_value(metrics.balanced_accuracy)])
        writer.writerow(["macro_f1", format_value(metrics.macro_f1)])
        writer.writerow(["weighted_f1", format_value(metrics.weighted_f1)])
        writer.writerow(["macro_precision", format_value(metrics.macro_precision)])
        writer.writerow(["macro_recall", format_value(metrics.macro_recall)])
        writer.writerow(["total_samples", str(metrics.total_samples)])
        for note in metrics.notes:
            writer.writerow(["note", note])
    return path
