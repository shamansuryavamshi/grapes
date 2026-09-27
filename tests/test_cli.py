"""CLI contract: argument validation, mutual exclusions, and exit codes.

The CLI is the primary interface, so a bad flag combination must fail loudly at
parse time rather than half-way through a training run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def grape():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import grape as module

    return module


def _parse(grape, argv):
    return grape.build_parser().parse_args(argv)


def _rejects(grape, argv):
    with pytest.raises(SystemExit) as excinfo:
        grape.build_parser().parse_args(argv)
    return excinfo.value.code


# ── basics ──────────────────────────────────────────────────────────────────
def test_help_exits_cleanly(grape, capsys):
    with pytest.raises(SystemExit) as excinfo:
        grape.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "usage" in out
    assert "irrelavant" in out, "the class list should be discoverable in --help"


def test_no_action_flags_means_train(grape):
    args = _parse(grape, [])
    for action in (
        "build_dataset",
        "evaluate",
        "replot",
        "resume",
        "serve",
        "leakage_check",
        "dataset_info",
        "check_dataset",
        "check_environment",
    ):
        assert getattr(args, action) is False, f"{action} should default to False"


def test_action_flags_round_trip(grape):
    assert _parse(grape, ["--evaluate"]).evaluate is True
    assert _parse(grape, ["--serve"]).serve is True
    assert _parse(grape, ["--resume"]).resume is True
    assert _parse(grape, ["--leakage-check"]).leakage_check is True
    assert _parse(grape, ["--dataset-info"]).dataset_info is True
    assert _parse(grape, ["--build-dataset"]).build_dataset is True


# ── numeric validation ──────────────────────────────────────────────────────
@pytest.mark.parametrize("value", ["0", "-1", "-40"])
def test_epochs_must_be_positive(grape, value):
    # "--epochs 0" would otherwise train nothing and still exit 0.
    assert _rejects(grape, ["--epochs", value]) != 0


@pytest.mark.parametrize("flag", ["--batch-size", "--image-size", "--aug-target"])
def test_counts_must_be_positive(grape, flag):
    assert _rejects(grape, [flag, "0"]) != 0
    assert _rejects(grape, [flag, "-2"]) != 0


def test_batch_size_zero_is_rejected(grape):
    assert _rejects(grape, ["--batch-size", "0"]) != 0


def test_workers_may_be_zero_but_not_negative(grape):
    assert _parse(grape, ["--workers", "0"]).workers == 0
    assert _rejects(grape, ["--workers", "-1"]) != 0


@pytest.mark.parametrize("flag", ["--val-split", "--test-split"])
def test_split_fractions_must_be_strictly_inside_the_unit_interval(grape, flag):
    for bad in ("0", "1", "1.5", "-0.2"):
        assert _rejects(grape, [flag, bad]) != 0, f"{flag} {bad} was accepted"
    assert getattr(_parse(grape, [flag, "0.2"]), flag.lstrip("-").replace("-", "_")) == 0.2


def test_val_and_test_splits_are_independent(grape):
    args = _parse(grape, ["--val-split", "0.1", "--test-split", "0.3"])
    assert args.val_split == 0.1
    assert args.test_split == 0.3


def test_probabilities_must_be_in_range(grape):
    for flag in ("--mixup-prob", "--label-smoothing"):
        assert _rejects(grape, [flag, "1.5"]) != 0
        assert _rejects(grape, [flag, "-0.1"]) != 0
        assert _parse(grape, [flag, "0.4"]).epochs is None


def test_learning_rate_must_be_positive(grape):
    assert _rejects(grape, ["--learning-rate", "0"]) != 0
    assert _rejects(grape, ["--learning-rate", "-1e-4"]) != 0


def test_weight_decay_may_be_zero(grape):
    assert _parse(grape, ["--weight-decay", "0"]).weight_decay == 0.0
    assert _rejects(grape, ["--weight-decay", "-0.1"]) != 0


def test_port_is_validated(grape):
    assert _parse(grape, ["--port", "8080"]).port == 8080
    for bad in ("0", "70000", "-1"):
        assert _rejects(grape, ["--port", bad]) != 0


def test_seed_is_configurable(grape):
    assert _parse(grape, ["--seed", "7"]).seed == 7
    assert _parse(grape, []).seed is None, "the default comes from the config, not argparse"


# ── flag combinations (validated in main) ───────────────────────────────────
@pytest.mark.parametrize(
    "argv",
    [
        ["--augment", "--skip-checks"],
        ["--augment", "--no-augment"],
    ],
)
def test_contradictory_flags_are_refused(grape, argv):
    # The combination used to silently skip the audit while still mutating the
    # dataset, which is how a leakage-free result gets quietly invalidated.
    with pytest.raises(SystemExit) as excinfo:
        grape.main(argv)
    assert excinfo.value.code != 0


# ── real subprocess smoke test ──────────────────────────────────────────────
def test_environment_check_runs_end_to_end(project_root):
    python = project_root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        pytest.skip("virtual environment not present")
    result = subprocess.run(
        [str(python), "grape.py", "--check-environment"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "irrelavant" in result.stdout.lower()


def test_bad_flag_exits_non_zero_end_to_end(project_root):
    python = project_root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        pytest.skip("virtual environment not present")
    result = subprocess.run(
        [str(python), "grape.py", "--epochs", "0"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "positive" in (result.stderr + result.stdout).lower()
