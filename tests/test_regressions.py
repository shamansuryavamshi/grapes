"""Regression tests for bugs that produced *plausible but wrong* output.

Each of these failed at some point while building the pipeline, and none of
them raised an error - they just quietly reported something untrue.
"""

from __future__ import annotations

import csv

import pytest

from iwnet.config import FINAL_CLASSES
from iwnet.data.build import _read_manifest_fingerprint, print_dataset_info


# ── the dataset fingerprint was never displayed ─────────────────────────────
def test_fingerprint_is_read_from_the_real_report_schema(cfg, capsys):
    """The report is written with columns ``metric,value``.

    Reading it as ``key,value`` returns None, so ``--dataset-info`` silently
    omitted the fingerprint that ties a run to its data.
    """
    results = cfg.paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    report = results / "dataset_report.csv"
    with open(report, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])  # the actual header
        writer.writerow(["seed", "42"])
        writer.writerow(["dataset_fingerprint_sha256", "deadbeef"])

    assert _read_manifest_fingerprint(cfg.paths.dataset_root) == "deadbeef"

    print_dataset_info(cfg)
    assert "deadbeef" in capsys.readouterr().out


def test_fingerprint_missing_is_not_an_error(cfg):
    assert _read_manifest_fingerprint(cfg.paths.dataset_root) is None


def test_fingerprint_is_none_when_the_report_is_corrupt(cfg):
    results = cfg.paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    (results / "dataset_report.csv").write_text("not,a,valid\nfingerprint,file\n")
    assert _read_manifest_fingerprint(cfg.paths.dataset_root) is None


# ── --evaluate used the *current* config, not the checkpoint's geometry ─────
def test_evaluate_checkpoint_uses_the_checkpoints_own_geometry(cfg, tmp_path, capsys):
    """A checkpoint trained at 64px must be evaluated at 64px.

    Reusing the current config would build a 224px model, fail the state_dict
    load with an opaque shape error - or if the shapes happened to match,
    evaluate under the wrong transform and report numbers never measured.
    """
    from iwnet.model.architecture import build_model
    from iwnet.model.checkpoint import save_checkpoint
    from iwnet.training.trainer import evaluate_checkpoint

    assert cfg.train.image_size == 64, "fixture builds a 64px model"

    model = build_model(cfg, pretrained=False)
    checkpoint = tmp_path / "geometry.pth"
    save_checkpoint(
        checkpoint,
        model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
        classes=list(FINAL_CLASSES),
        cfg=cfg,
        epoch=5,
        val_acc=0.5,
    )

    # Now change the *current* config to a different geometry. The evaluation
    # must follow the checkpoint, not the config.
    cfg.train.image_size = 224
    cfg.model.backbone = "efficientnet_b3"

    result = evaluate_checkpoint(cfg, checkpoint)
    assert result is not None, "evaluate_checkpoint returned nothing"
    out = capsys.readouterr().out
    assert "image_size=64" in out, (
        "evaluation used the current config's image size instead of the checkpoint's"
    )
    assert "efficientnet_b0" in out, (
        "evaluation used the current config's backbone instead of the checkpoint's"
    )


def test_evaluate_checkpoint_reports_a_missing_file(cfg, tmp_path):
    from iwnet.training.trainer import evaluate_checkpoint

    assert evaluate_checkpoint(cfg, tmp_path / "nope.pth") is None


# ── /api/model reported nulls for a checkpoint that had the data ───────────
def test_api_model_does_not_report_nulls_for_a_real_checkpoint(api_with_model):
    """It asked for val_accuracy/dataset_seed/versions.

    The checkpoint stores val_acc/seed/*_version, so every one of those came
    back null - which reads as "nothing was recorded" rather than as the bug it
    was, on a model that had trained fine.
    """
    body = api_with_model.get("/api/model").json()

    assert body["val_accuracy"] == pytest.approx(0.5), (
        "val accuracy is missing from the checkpoint metadata"
    )
    assert body["dataset_seed"] == 42
    assert body["image_size"] == 64
    assert body["backbone"] == "efficientnet_b0"
    assert body["versions"]["torch"], "library versions should be reported"
    assert body["load_seconds"] > 0


def test_predictor_exposes_image_size(cfg, tmp_path):
    from iwnet.inference.predictor import Predictor
    from iwnet.model.architecture import build_model
    from iwnet.model.checkpoint import save_checkpoint

    model = build_model(cfg, pretrained=False)
    path = cfg.paths.best_checkpoint
    save_checkpoint(
        path,
        model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
        classes=list(FINAL_CLASSES),
        cfg=cfg,
        epoch=1,
        val_acc=0.5,
    )
    predictor = Predictor(cfg, path).load()
    assert predictor.image_size == 64
    assert predictor.metadata["val_acc"] == pytest.approx(0.5)
    assert predictor.metadata["seed"] == 42
