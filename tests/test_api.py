"""API behaviour, including the degraded (no-checkpoint) state.

The service must never crash on startup just because no model has been trained
yet, and it must not let a caller read arbitrary files off disk.
"""

from __future__ import annotations

import pytest

from conftest import png_bytes
from iwnet.config import FINAL_CLASSES



# â”€â”€ degraded state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_health_is_200_without_a_model(api_without_model):
    response = api_without_model.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded", "no_model"}


def test_health_reports_the_missing_checkpoint(api_without_model):
    body = api_without_model.get("/api/health").json()
    assert body.get("model_loaded") is False
    assert body.get("detail") or body.get("message")


def test_model_endpoint_is_503_without_a_model(api_without_model):
    response = api_without_model.get("/api/model")
    assert response.status_code == 503
    assert "train" in response.json()["detail"].lower()


def test_predict_is_503_without_a_model(api_without_model):
    response = api_without_model.post(
        "/api/predict", files={"file": ("leaf.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 503


def test_classes_work_without_a_model(api_without_model):
    """Class metadata is static, so it should not depend on a trained model."""
    response = api_without_model.get("/api/classes")
    assert response.status_code == 200
    names = [row["name"] for row in response.json()]
    assert names == list(FINAL_CLASSES)


def test_static_frontend_is_served(api_without_model):
    assert api_without_model.get("/").status_code == 200
    assert api_without_model.get("/app.js").status_code == 200
    assert api_without_model.get("/styles.css").status_code == 200


# â”€â”€ working model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_health_is_200_with_a_model(api_with_model):
    body = api_with_model.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_endpoint_describes_the_checkpoint(api_with_model):
    body = api_with_model.get("/api/model").json()
    assert body["classes"] == list(FINAL_CLASSES)
    assert body["num_classes"] == len(FINAL_CLASSES)
    assert body["epoch"] == 1
    assert body["dataset_fingerprint_sha256"] is None or isinstance(
        body["dataset_fingerprint_sha256"], str
    )
    assert "IWNET" in body["architecture"]


def test_predict_returns_a_valid_class(api_with_model):
    response = api_with_model.post(
        "/api/predict", files={"file": ("leaf.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in FINAL_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert [row["name"] for row in body["probabilities"]] == list(FINAL_CLASSES)
    assert sum(row["probability"] for row in body["probabilities"]) == pytest.approx(
        1.0, abs=1e-3
    )


def test_predict_declares_confidence_is_not_calibrated(api_with_model):
    body = api_with_model.post(
        "/api/predict", files={"file": ("leaf.png", png_bytes(), "image/png")}
    ).json()
    text = (body.get("confidence_note") or "").lower()
    assert "not calibrated" in text, (
        "the API must not present softmax output as diagnostic probability"
    )


@pytest.mark.parametrize(
    "payload,content_type",
    [
        (b"", "image/png"),
        (b"not an image at all", "image/png"),
        (b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, "image/png"),
    ],
)
def test_predict_rejects_corrupt_uploads(api_with_model, payload, content_type):
    response = api_with_model.post(
        "/api/predict", files={"file": ("leaf.png", payload, content_type)}
    )
    assert response.status_code in {400, 415, 422}, (
        f"a corrupt upload returned {response.status_code}"
    )


def test_predict_requires_a_file(api_with_model):
    assert api_with_model.post("/api/predict").status_code in {400, 422}


def test_predict_rejects_an_oversized_upload(api_with_model, monkeypatch):
    import iwnet.api.app as app_module

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)
    response = api_with_model.post(
        "/api/predict", files={"file": ("big.png", png_bytes(size=(400, 400)), "image/png")}
    )
    assert response.status_code in {413, 400, 422}


def test_sample_image_rejects_path_traversal(api_with_model):
    for attempt in ("../../secrets", "..%2F..%2Fsecrets", "..\\..\\secrets"):
        response = api_with_model.get(f"/api/samples/Healthy/{attempt}")
        assert response.status_code in {400, 404}, (
            f"path traversal was served: {attempt} -> {response.status_code}"
        )


def test_sample_image_rejects_an_unknown_class(api_with_model):
    response = api_with_model.get("/api/samples/NotAClass/x.jpg")
    assert response.status_code in {400, 404}


def test_openapi_schema_is_valid(api_with_model):
    schema = api_with_model.get("/openapi.json")
    assert schema.status_code == 200
    body = schema.json()
    for route in ("/api/health", "/api/predict", "/api/classes", "/api/model"):
        assert route in body["paths"], f"{route} missing from the OpenAPI schema"

