#!/usr/bin/env python
"""IWNET grape leaf disease classification - command-line entry point.

This file is intentionally a **thin dispatcher**. All real work lives in the
``iwnet`` package so it can be imported and tested; nothing here duplicates
logic. See ``README.md`` for the full command reference.

Quick start
-----------
    python grape.py --check-environment
    python grape.py --build-dataset
    python grape.py --dataset-info
    python grape.py
    python grape.py --serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python grape.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 with replacement, so printing never raises.

    Windows consoles default to a legacy code page (cp1252). Any non-ASCII
    character in a report - box-drawing rules, a degree sign, an em dash - then
    raises ``UnicodeEncodeError`` and kills the process. That is a poor way to
    lose a finished training run, so undecodable characters degrade to '?'
    instead.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - unusual stream
            pass


_force_utf8_output()

from iwnet.config import FINAL_CLASSES, config_from_cli_args  # noqa: E402
from iwnet.utils import configure_logging, get_logger  # noqa: E402

log = get_logger("grape")

BANNER = r"""
 _       ____  __    _   _   _____ ____   _____ ____
| |     / _ \ \  \  | \ | | |_   _|  _ \ | ____|  _ \
| |    | | | | \  \  |  \| |   | | | | ||  _| | |_) |
| |___ | |_| |  \  \ | |\  |   | | |_| || |___|  _ <
 \____/ \___/   \__\_| \_| |___|_|  \___/ |_____|_| \_\
"""


# ── argparse validators ────────────────────────────────────────────────────
# These exist so a typo fails at parse time with a message that names the flag,
# instead of silently doing nothing (e.g. "--epochs 0" reporting a successful
# run) or crashing somewhere deep in the DataLoader.


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _nonneg_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or greater, got {value}")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def _nonneg_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or greater, got {value}")
    return value


def _unit_float(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def _fraction(raw: str) -> float:
    # Exclusive at both ends: a 0.0 or 1.0 split would leave one side empty.
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(
            f"must be strictly between 0 and 1, got {value}"
        )
    return value


def _port(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"must be a valid TCP port, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grape.py",
        description=(
            "Train, evaluate and serve the IWNET grape leaf disease classifier "
            f"({len(FINAL_CLASSES)} classes)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Classes ({len(FINAL_CLASSES)}):\n"
            + "".join(f"  {index}. {name}\n" for index, name in enumerate(FINAL_CLASSES))
            + "\n"
            "  'Irrelavant' is intentionally misspelled. It is the name used by the\n"
            "  source folders and is baked into every trained checkpoint.\n"
            "\n"
            "Typical workflow:\n"
            "  python grape.py --check-environment\n"
            "  python grape.py --build-dataset\n"
            "  python grape.py --dataset-info\n"
            "  python grape.py --leakage-check\n"
            "  python grape.py                       # train, then evaluate\n"
            "  python grape.py --serve               # web UI on :8000\n"
        ),
    )

    # ── actions (mutually exclusive) ────────────────────────────────────────
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-environment", action="store_true",
                         help="Verify Python, PyTorch, CUDA and dependencies. Exits non-zero on failure.")
    actions.add_argument("--build-dataset", action="store_true",
                         help="Rebuild the generated dataset from the read-only source folders.")
    actions.add_argument("--dataset-info", action="store_true",
                         help="Show per-class counts for the generated dataset and its fingerprint.")
    actions.add_argument("--check-dataset", action="store_true",
                         help="Validate the dataset: class coverage and full image decode.")
    actions.add_argument("--leakage-check", action="store_true",
                         help="Audit train/test for exact, cross-class and near duplicates.")
    actions.add_argument("--predict", metavar="IMAGE",
                         help="Classify a single image and print the result.")
    actions.add_argument("--evaluate", action="store_true",
                         help="Evaluate an existing checkpoint on the test split (no training).")
    actions.add_argument("--replot", action="store_true",
                         help="Regenerate plots and CSV reports from a saved evaluation.")
    actions.add_argument("--resume", action="store_true",
                         help="Continue training from last_iwnet.pth.")
    actions.add_argument("--serve", action="store_true",
                         help="Run the FastAPI server and the web UI.")

    # ── configuration ──────────────────────────────────────────────────────
    parser.add_argument("--config", metavar="FILE", default=None,
                        help="YAML config file (default: config.yaml if present).")
    parser.add_argument("--set", dest="overrides", metavar="KEY=VALUE", action="append", default=[],
                        help="Override any config value, e.g. --set train.epochs=30.")

    # ── training overrides ──────────────────────────────────────────────────
    # Every numeric flag is validated at parse time. Without this, "--epochs 0"
    # trains nothing and still reports success, and "--batch-size 0" fails deep
    # inside a DataLoader with a message that does not mention the CLI at all.
    parser.add_argument("--epochs", type=_positive_int, default=None,
                        help="Number of epochs.")
    parser.add_argument("--batch-size", type=_positive_int, default=None,
                        help="Batch size (lower it on small GPUs).")
    parser.add_argument("--image-size", type=_positive_int, default=None,
                        help="Training/inference image size.")
    parser.add_argument("--learning-rate", type=_positive_float, default=None,
                        help="Base learning rate.")
    parser.add_argument("--weight-decay", type=_nonneg_float, default=None,
                        help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=_nonneg_int, default=None,
                        help="Linear-warmup epochs.")
    parser.add_argument("--unfreeze-epoch", type=_nonneg_int, default=None,
                        help="Epoch at which the backbone unfreezes.")
    parser.add_argument("--unfreeze-stages", type=_nonneg_int, default=None,
                        help="Backbone stages to unfreeze.")
    parser.add_argument("--early-stop-patience", type=_nonneg_int, default=None,
                        help="Early-stopping patience in epochs.")
    parser.add_argument("--mixup-alpha", type=_nonneg_float, default=None,
                        help="MixUp alpha (0 disables).")
    parser.add_argument("--mixup-prob", type=_unit_float, default=None,
                        help="Probability of applying MixUp.")
    parser.add_argument("--label-smoothing", type=_unit_float, default=None,
                        help="Label smoothing factor.")
    parser.add_argument("--val-split", type=_fraction, default=None,
                        help="Validation fraction carved from train (0-1).")
    parser.add_argument("--test-split", type=_fraction, default=None,
                        help="Test fraction at build time (0-1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default 42).")
    parser.add_argument("--aug-target", type=_positive_int, default=None,
                        help="Offline augmentation target per class.")

    # ── data loaders / augmentation ─────────────────────────────────────────
    parser.add_argument("--workers", type=_nonneg_int, default=None,
                        help="DataLoader worker processes.")
    parser.add_argument("--augment", action="store_true",
                        help="Offline-augment minority classes in TRAIN only (before splitting).")
    parser.add_argument("--no-augment", action="store_true", help="Disable offline augmentation.")
    parser.add_argument("--aug-only-train", action="store_true", default=True,
                        help="Kept for compatibility; offline augmentation is always train-only.")

    # ── runtime ─────────────────────────────────────────────────────────────
    parser.add_argument("--device", default=None,
                        help="Device override: cpu, cuda, cuda:0, mps. Default: auto.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download ImageNet weights for the backbone.")
    parser.add_argument("--no-group-splits", action="store_true",
                        help="Disable group-aware splitting (allows near-duplicate leakage; for reproducing old runs only).")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip dataset validation and the leakage audit before training. Never combine with --augment.")
    parser.add_argument("--allow-leakage", action="store_true",
                        help="Proceed even if the leakage audit finds blocking duplicates.")
    parser.add_argument("--quiet", action="store_true", help="Reduce log verbosity.")

    # ── server ──────────────────────────────────────────────────────────────
    parser.add_argument("--host", default=None, help="API bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=_port, default=None, help="API port (default 8000).")
    parser.add_argument("--reload", action="store_true", help="Auto-reload the server on code changes (development).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(level="WARNING" if args.quiet else "INFO")

    # `--augment` and `--skip-checks` together used to silently skip the audit
    # while still mutating the dataset. Refuse the combination outright.
    if args.augment and args.skip_checks:
        parser.error(
            "--augment cannot be combined with --skip-checks. Offline augmentation "
            "rewrites the training split, so the dataset and leakage checks must run "
            "afterwards. Drop --skip-checks, or run --augment separately."
        )
    if args.augment and args.no_augment:
        parser.error("--augment and --no-augment are mutually exclusive.")

    cfg = config_from_cli_args(args)
    if args.overrides:
        from iwnet.config import apply_overrides

        apply_overrides(cfg, args.overrides)
    if args.workers is not None:
        cfg.data.num_workers = args.workers
    if args.device:
        cfg.train.device = args.device
    if args.no_group_splits:
        cfg.data.group_splits = False
    if args.allow_leakage:
        cfg.leakage.strict = False
    if args.augment:
        cfg.data.aug_enabled = True
    if args.no_augment:
        cfg.data.aug_enabled = False

    print(BANNER)
    print(f"  {len(FINAL_CLASSES)} classes: {', '.join(FINAL_CLASSES)}\n")

    # ── dispatch ────────────────────────────────────────────────────────────
    if args.check_environment:
        from iwnet.utils import environment_report

        report = environment_report()
        print("  ENVIRONMENT")
        print("  " + "-" * 66)
        for key, value in report.items():
            if isinstance(value, (dict, list)):
                continue
            print(f"    {key:<24} {value}")
        print()
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key:<24} {sub_value}")
        print()
        problems = [k for k, v in report.items() if k == "torch" and v in (None, "", "missing")]
        if problems:
            print("  [FAIL] Required packages are missing.\n")
            return 1
        print("  [OK] Environment looks usable.\n")
        return 0

    if args.build_dataset:
        from iwnet.data.build import build_dataset_from_sources

        return 0 if build_dataset_from_sources(cfg).ok else 1

    if args.dataset_info:
        from iwnet.data.build import print_dataset_info

        print_dataset_info(cfg)
        return 0

    if args.check_dataset:
        from iwnet.data.validation import print_report, validate_dataset

        report = validate_dataset(cfg, deep=True)
        print_report(report)
        return 0 if report.ok else 1

    if args.leakage_check:
        from iwnet.data.leakage import run_leakage_check

        report = run_leakage_check(cfg, allow=args.allow_leakage)
        return 0 if (report.blocking == [] or not cfg.leakage.strict) else 1

    if args.predict:
        return _cmd_predict(cfg, args.predict)

    if args.evaluate:
        from iwnet.training.trainer import evaluate_checkpoint

        result = evaluate_checkpoint(cfg)
        return 0 if result is not None else 1

    if args.replot:
        from iwnet.training.plots import replot_from_results

        return replot_from_results(cfg)

    if args.resume:
        from iwnet.training.trainer import run_training

        return 0 if run_training(cfg, resume=True) else 1

    if args.serve:
        return _cmd_serve(cfg, args)

    # Default action: train.
    from iwnet.training.trainer import run_training

    return 0 if run_training(cfg, resume=False) else 1


def _cmd_predict(cfg, image: str) -> int:
    from iwnet.inference.predictor import InferenceError, get_predictor

    try:
        prediction = get_predictor(cfg).predict_path(image)
    except InferenceError as exc:
        print(f"\n[ERROR] {exc}\n")
        return 1

    print("=" * 68)
    print("  PREDICTION")
    print("=" * 68)
    print(f"  Image            : {Path(image).name}")
    print(f"  Predicted class  : {prediction.class_info.get('label', prediction.predicted_class)}")
    print(f"  Model confidence : {prediction.confidence * 100:.2f}%")
    if prediction.is_rejection:
        print("  -> Rejection class: this does not look like a grape leaf.")
    print()
    print("  Class probabilities (softmax, uncalibrated):")
    for name, value in sorted(
        prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True
    ):
        bar = "#" * int(value * 40)
        print(f"    {name:<16} {value * 100:6.2f}%  {bar}")
    if prediction.class_info.get("note"):
        print(f"\n  Note: {prediction.class_info['note']}")
    print(
        "\n  This is a research classifier, not a diagnostic device. "
        "Confidence is not calibrated."
    )
    print("=" * 68)
    return 0


def _cmd_serve(cfg, args) -> int:
    host = args.host or cfg.api.host
    port = args.port or cfg.api.port
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is not installed. Run: pip install -r requirements.txt")
        return 1

    print(f"  Serving on http://{host}:{port}")
    print(f"  API docs          http://{host}:{port}/docs")
    print(f"  Datasets available: {cfg.paths.dataset_root}")
    print("  Press Ctrl+C to stop.\n")
    uvicorn.run("iwnet.api.app:app", host=host, port=port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
