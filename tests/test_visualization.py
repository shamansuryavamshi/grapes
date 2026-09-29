"""Tests for the inference-visualization feature.

Two rules govern everything in this file:

1. **The prediction must not change.** Capturing intermediates is observation, not
   intervention, so ``?visualize=true`` and the default response must carry
   byte-identical class probabilities. That is asserted directly rather than
   assumed.
2. **A visualization failure is never a prediction failure.** The optional stage
   is wrapped so it cannot take a successful prediction down with it.

Nothing here asserts a specific disease for a specific image. The fixture model is
randomly initialised, so any assertion about *which* class wins would be testing
the seed, not the API.
"""

from __future__ import annotations

import base64
import json
import re

import pytest

from conftest import png_bytes
from iwnet.config import FINAL_CLASSES

STAGE_KEYS = (
    "original",
    "resized",
    "normalized",
    "features",
    "channel_attention",
    "spatial_attention",
    "multi_scale",
    "probabilities",
)


def _upload(client, **params):
    return client.post("/api/predict", files={"file": ("leaf.png", png_bytes(), "image/png")}, **params)


def _png_size(data_url: str):
    """Read width/height straight out of the PNG header of a data URL."""
    import struct

    assert data_url.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", raw[16:24])
    return int(width), int(height)


def _is_not_blank(data_url: str) -> bool:
    """A rendered image that carries actual structure, not a flat fill."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float(array.std()) > 5.0


# ── 1-4: every existing prediction input still works ─────────────────────────
def test_prediction_without_visualize_still_works(api_with_model):
    response = _upload(api_with_model)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in FINAL_CLASSES
    assert isinstance(body["confidence"], float)
    assert len(body["probabilities"]) == 7


def test_multipart_prediction_still_works(api_with_model):
    response = api_with_model.post(
        "/api/predict", files={"file": ("leaf.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in FINAL_CLASSES


def test_raw_body_prediction_still_works(api_with_model):
    response = api_with_model.post(
        "/api/predict", content=png_bytes(), headers={"content-type": "image/png"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in FINAL_CLASSES


def test_image_path_prediction_still_works(real_api_with_model):
    entries = real_api_with_model.get("/api/samples", params={"per_class": 1}).json()
    assert entries
    for entry in entries:
        response = real_api_with_model.post(
            "/api/predict", params={"image_path": entry["path"]}
        )
        assert response.status_code == 200
        assert response.json()["prediction"] in FINAL_CLASSES


# ── 5: the disabled path returns the pre-existing response shape ────────────
def test_visualization_disabled_returns_existing_response_structure(api_with_model):
    body = _upload(api_with_model).json()
    for key in (
        "prediction",
        "label",
        "confidence",
        "is_rejection",
        "description",
        "note",
        "probabilities",
        "timing_ms",
        "model_trained_at_utc",
        "dataset_fingerprint_sha256",
        "confidence_note",
    ):
        assert key in body, f"pre-existing field {key} disappeared"
    # Absent, not null-and-present: the default response is unchanged in spirit.
    assert body.get("visualization") is None


# ── 6: the enabled path adds diagnostics without dropping anything ─────────
def test_visualization_enabled_returns_diagnostic_fields(api_with_model):
    response = _upload(api_with_model, params={"visualize": "true"})
    assert response.status_code == 200
    body = response.json()

    # Every original field survived.
    assert body["prediction"] in FINAL_CLASSES
    assert isinstance(body["confidence"], float)
    assert len(body["probabilities"]) == 7
    assert "preprocessing" in body["timing_ms"] and "inference" in body["timing_ms"]

    viz = body["visualization"]
    assert viz["available"] is True
    assert viz["engine"] == "activation-capture"
    for key in STAGE_KEYS:
        assert key in viz["stages"], f"stage {key} missing"
        stage = viz["stages"][key]
        assert isinstance(stage["title"], str) and stage["title"]
        assert isinstance(stage["caption"], str) and stage["caption"]
    assert viz["disclaimer"]


# ── 7: original + resized ───────────────────────────────────────────────────
def test_original_and_resized_stages(api_with_model, cfg):
    viz = _upload(api_with_model, params={"visualize": "true"}).json()["visualization"]

    original = viz["stages"]["original"]
    assert original["available"] is True
    assert _png_size(original["image"]) == (64, 64)  # png_bytes() default
    assert original["width"] == 64 and original["height"] == 64

    resized = viz["stages"]["resized"]
    assert resized["available"] is True
    # The production transform is Resize(round(size*1.14)) then CenterCrop(size).
    assert (_png_size(resized["image"]), resized["width"], resized["height"]) == (
        (cfg.train.image_size, cfg.train.image_size),
        cfg.train.image_size,
        cfg.train.image_size,
    )
    assert str(cfg.train.image_size) in resized["title"]
    assert _is_not_blank(resized["image"])


def test_image_path_mode_omits_the_original_stage(real_api_with_model):
    entry = real_api_with_model.get("/api/samples", params={"per_class": 1}).json()[0]
    body = real_api_with_model.post(
        "/api/predict",
        params={"image_path": entry["path"], "visualize": "true"},
    ).json()
    original = body["visualization"]["stages"]["original"]
    assert original["available"] is False
    assert original["image"] is None
    # Everything else is still produced.
    assert body["visualization"]["stages"]["resized"]["available"] is True


# ── 8: normalized ───────────────────────────────────────────────────────────
def test_normalized_stage(api_with_model):
    viz = _upload(api_with_model, params={"visualize": "true"}).json()["visualization"]
    stage = viz["stages"]["normalized"]
    assert stage["available"] is True
    assert _png_size(stage["image"])[0] == _png_size(stage["inverse_image"])[0]
    assert _is_not_blank(stage["image"])
    assert _is_not_blank(stage["inverse_image"])
    # The inverse is display-only and says so.
    assert "never fed back into the model" in stage["inverse_caption"].lower()


# ── 9: spatial attention ────────────────────────────────────────────────────
def test_spatial_attention_stage(api_with_model):
    viz = _upload(api_with_model, params={"visualize": "true"}).json()["visualization"]
    stage = viz["stages"]["spatial_attention"]
    assert stage["available"] is True
    assert _is_not_blank(stage["image"])
    assert _is_not_blank(stage["montage"])
    assert len(stage["maps"]) == 3
    for entry in stage["maps"]:
        assert entry["height"] > 0 and entry["width"] > 0
        assert len(entry["values"]) == min(entry["height"], 32) * min(entry["width"], 32)
        assert all(0.0 <= v <= 1.0 for v in entry["values"])
    # Wording must stay descriptive, not diagnostic.
    caption = stage["caption"].lower()
    assert "spatial weighting" in caption
    assert "not a claim" in caption


# ── 10: channel attention ───────────────────────────────────────────────────
def test_channel_attention_stage(api_with_model):
    viz = _upload(api_with_model, params={"visualize": "true"}).json()["visualization"]
    stage = viz["stages"]["channel_attention"]
    assert stage["available"] is True
    assert len(stage["stages"]) == 3
    total = 0
    for entry in stage["stages"]:
        assert len(entry["weights"]) == entry["channels"]
        assert all(0.0 <= w <= 1.0 for w in entry["weights"])
        assert len(entry["top_channels"]) == len(entry["top_weights"]) <= 12
        assert entry["top_weights"] == sorted(entry["top_weights"], reverse=True)
        total += entry["channels"]
    assert total > 0
    assert "not that it mattered" in stage["caption"].lower()


def test_feature_representation_stage(api_with_model):
    viz = _upload(api_with_model, params={"visualize": "true"}).json()["visualization"]
    stage = viz["stages"]["features"]
    assert stage["available"] is True
    assert _is_not_blank(stage["image"])
    assert len(stage["stages"]) == 3
    for entry in stage["stages"]:
        assert len(entry["top_channels"]) == 8
        assert len(entry["top_channels"]) == len(set(entry["top_channels"]))
        assert entry["channels"] > 0 and entry["height"] > 0
    assert "not a saliency map" in stage["caption"].lower()


def test_multi_scale_stage_reports_real_weights(api_with_model):
    viz = _upload(api_with_model, params={"visualize": "true"}).json()["visualization"]
    stage = viz["stages"]["multi_scale"]
    assert stage["available"] is True
    assert len(stage["scales"]) == 3
    assert stage["weight_sum"] == pytest.approx(1.0, abs=1e-4)
    assert all(0.0 <= s["weight"] <= 1.0 for s in stage["scales"])
    # Weights come from the model, so they must not all be identical or fixed.
    assert len({round(s["weight"], 6) for s in stage["scales"]}) > 1
    assert _is_not_blank(stage["image"])
    assert len(stage["iw_gates"]) == 3


# ── 11: the headline invariant ──────────────────────────────────────────────
def test_probabilities_identical_with_and_without_visualization(api_with_model):
    off = _upload(api_with_model).json()
    on = _upload(api_with_model, params={"visualize": "true"}).json()

    assert off["prediction"] == on["prediction"]
    assert off["confidence"] == on["confidence"]
    assert off["probabilities"] == on["probabilities"]
    assert off["is_rejection"] == on["is_rejection"]

    # And the copy embedded in the visualization must match too.
    embedded = on["visualization"]["stages"]["probabilities"]["rows"]
    by_name = {row["name"]: row["probability"] for row in embedded}
    assert by_name == {p["name"]: p["probability"] for p in off["probabilities"]}
    assert by_name[on["prediction"]] == max(by_name.values())
    # Exactly one predicted row, and it is the response's prediction.
    assert [row["name"] for row in embedded if row["predicted"]] == [on["prediction"]]


def test_capture_does_not_change_the_predictor_output(real_config):
    """The invariant at the library level, on the real checkpoint."""
    from PIL import Image

    from iwnet.inference.predictor import Predictor

    dataset = real_config.paths.dataset_root
    sample = sorted(dataset.rglob("*.jpg"))[0]
    predictor = Predictor(real_config).load()
    with Image.open(sample) as handle:
        image = handle.convert("RGB")

    plain = predictor.predict_pil(image)
    traced, trace, tensor = predictor.predict_pil_traced(image)

    assert plain.predicted_class == traced.predicted_class
    assert plain.confidence == traced.confidence
    assert plain.probabilities == traced.probabilities
    assert trace.is_complete
    assert tensor.shape[0] == 1 and tensor.shape[1] == 3


def test_capture_leaves_no_hooks_on_the_model(real_config):
    from PIL import Image

    from iwnet.inference.predictor import Predictor

    dataset = real_config.paths.dataset_root
    predictor = Predictor(real_config).load()
    with Image.open(sorted(dataset.rglob("*.jpg"))[0]) as handle:
        image = handle.convert("RGB")

    predictor.predict_pil_traced(image)
    hooked = [
        name for name, module in predictor._model.named_modules() if len(module._forward_hooks)
    ]
    assert hooked == [], f"hooks left installed on {hooked}"

    # A later plain prediction must still work and be unchanged.
    assert predictor.predict_pil(image).predicted_class


def test_capture_uses_one_model_instance_only(real_config):
    from PIL import Image

    from iwnet.inference.predictor import Predictor

    predictor = Predictor(real_config).load()
    ids_before = {id(p) for p in predictor._model.parameters()}
    with Image.open(sorted(real_config.paths.dataset_root.rglob("*.jpg"))[0]) as handle:
        predictor.predict_pil_traced(handle.convert("RGB"))
    assert {id(p) for p in predictor._model.parameters()} == ids_before


# ── 12: visualization failure must not fail the prediction ──────────────────
def test_visualization_failure_does_not_fail_prediction(api_with_model, monkeypatch):
    import iwnet.inference.visualize as viz_module

    def boom(**_kwargs):
        raise RuntimeError("synthetic visualization failure")

    monkeypatch.setattr(viz_module, "build_visualization", boom)

    response = _upload(api_with_model, params={"visualize": "true"})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in FINAL_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 7
    assert body["visualization"]["available"] is False
    assert "unavailable" in body["visualization"]["message"].lower()
    # The failure reason must not be an internal traceback.
    assert "synthetic visualization failure" not in json.dumps(body["visualization"])


def test_visualization_failure_still_returns_the_same_prediction(api_with_model, monkeypatch):
    import iwnet.inference.visualize as viz_module

    before = _upload(api_with_model).json()
    monkeypatch.setattr(
        viz_module, "build_visualization", lambda **_: (_ for _ in ()).throw(OSError("nope"))
    )
    after = _upload(api_with_model, params={"visualize": "true"}).json()
    assert before["prediction"] == after["prediction"]
    assert before["probabilities"] == after["probabilities"]


# ── 13: traversal protection unchanged, with and without visualize ──────────
@pytest.mark.parametrize(
    "attempt",
    ["../secrets", "..%2F..%2Fconfig.yaml", "..\\..\\requirements.txt", "/etc/passwd"],
)
def test_traversal_protection_still_intact_with_visualize(real_api_with_model, attempt):
    response = real_api_with_model.post(
        "/api/predict", params={"image_path": attempt, "visualize": "true"}
    )
    assert response.status_code in {400, 403, 404}
    assert response.status_code != 500


# ── 14: sample gallery untouched ────────────────────────────────────────────
def test_sample_gallery_still_functions(real_api_with_model):
    response = real_api_with_model.get("/api/samples", params={"per_class": 1})
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 7
    for entry in entries:
        assert entry["class"] in FINAL_CLASSES
        served = real_api_with_model.get(entry["url"])
        assert served.status_code == 200
        assert served.headers["content-type"].startswith("image/")


# ── security: the payload must not leak anything about the host ─────────────
def test_visualization_payload_leaks_no_paths_or_secrets(api_with_model):
    payload = json.dumps(_upload(api_with_model, params={"visualize": "true"}).json())

    assert not re.search(r"[A-Za-z]:[\\/]", payload), "absolute Windows path in payload"
    assert not re.search(r"/(home|Users|root|tmp)/", payload), "absolute POSIX path in payload"
    for needle in ("suryavamshi", "site-packages", "Balanced_", "checkpoint", ".pth"):
        assert needle not in payload, f"payload leaked {needle!r}"
    # Visualizations must be inline data URLs, not server-relative file paths.
    for url in re.findall(r'"image":\s*"([^"]*)"', payload):
        assert url.startswith("data:image/png;base64,")


def test_visualization_response_stays_reasonably_small(api_with_model):
    off = len(_upload(api_with_model).content)
    on = len(_upload(api_with_model, params={"visualize": "true"}).content)
    assert on < 6 * 1024 * 1024, "visualization payload is unexpectedly large"
    assert on > off  # it really did carry something extra


def test_trace_release_clears_the_captured_tensors(real_config):
    from PIL import Image

    from iwnet.inference.predictor import Predictor

    predictor = Predictor(real_config).load()
    with Image.open(sorted(real_config.paths.dataset_root.rglob("*.jpg"))[0]) as handle:
        _, trace, _ = predictor.predict_pil_traced(handle.convert("RGB"))
    trace.release()
    assert trace.scale_weights is None
    assert all(item is None for item in trace.projected)
    assert all(item is None for item in trace.spatial_attention)
    assert all(item is None for item in trace.channel_attention)
