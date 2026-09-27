"""Training loop, metrics and plotting."""

from iwnet.training.metrics import (
    ClassificationMetrics,
    PerClassMetric,
    compute_metrics,
    write_classification_report_csv,
)
from iwnet.training.plots import (
    plot_accuracy_curve,
    plot_class_confidence,
    plot_confusion_matrix,
    plot_dataset_distribution,
    plot_loss_curve,
    plot_lr_schedule,
    plot_per_class_accuracy,
    plot_precision_recall_f1,
    plot_roc_auc,
    plot_sample_predictions,
)
from iwnet.training.trainer import (
    EpochResult,
    EvalResult,
    TrainingFailure,
    WarmupCosineScheduler,
    evaluate,
    run_evaluation,
    run_training,
)

__all__ = [
    "ClassificationMetrics",
    "PerClassMetric",
    "compute_metrics",
    "write_classification_report_csv",
    "TrainingFailure",
    "WarmupCosineScheduler",
    "EpochResult",
    "EvalResult",
    "run_training",
    "run_evaluation",
    "evaluate",
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_lr_schedule",
    "plot_dataset_distribution",
    "plot_confusion_matrix",
    "plot_per_class_accuracy",
    "plot_precision_recall_f1",
    "plot_roc_auc",
    "plot_sample_predictions",
    "plot_class_confidence",
]
