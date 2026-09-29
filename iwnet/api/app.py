"""FastAPI service exposing the trained IWNET model.

Endpoints
---------
``GET  /api/health``      liveness + model-loaded state (never fails on a missing model)
``GET  /api/model``       model metadata, class list and training provenance
``GET  /api/classes``     class descriptions for the UI
``POST /api/predict``     predict from a multipart upload, a raw body, or a dataset path
``GET  /api/samples``     a few labelled dataset images for the "try it" panel
``GET  /``                the single-page UI

Deliberate choices
------------------
* The model is loaded **once** at startup (lifespan), not per request.
* Startup does **not** crash when the model is missing. The service comes up,
  ``/api/health`` reports ``model_loaded: false``, and prediction endpoints return
  a clear 503. That makes the deployment diagnosable instead of crash-looping.
* Uploaded bytes are size-limited and fully decoded before use, so a malicious
  or malformed upload cannot exhaust memory.
* Internal errors are logged with detail but returned to clients as short,
  non-leaking messages.
"""

from __future__ import annotations

import io
import logging
import os
import random
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from iwnet.classes import class_order
from iwnet.config import FINAL_CLASSES, Config, load_config
from iwnet.inference.predictor import InferenceError, Predictor, get_predictor, reset_predictor
from iwnet.model.checkpoint import CheckpointError
from iwnet.utils import get_logger, set_seed

log = get_logger("api.app")
api_log = logging.getLogger("uvicorn.error")

#: Hard cap on a single upload. 10 MB is far above a 224x224 JPEG.
MAX_UPLOAD_BYTES = int(os.environ.get("IWNET_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
#: Vanilla-JS single-page UI. No build step, so it works from a clean checkout.
FRONTEND_DIR = os.environ.get(
    "IWNET_FRONTEND_DIR",
    str(load_config().paths.frontend_dir),
)

_CONFIG: Config | None = None
_STARTUP_ERROR: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model exactly once, on startup."""
    global _CONFIG, _STARTUP_ERROR
    cfg = load_config()
    _CONFIG = cfg
    set_seed(cfg.data.seed)
    try:
        predictor = get_predictor(cfg)
        api_log.info(
            "IWNET model ready: %d classes, device=%s, loaded in %.2fs",
            len(predictor.classes), predictor._device, predictor.load_seconds,
        )
    except (CheckpointError, InferenceError, OSError) as exc:
        _STARTUP_ERROR = str(exc)
        log.error("Model not loaded at startup: %s", exc)
        api_log.warning("Starting WITHOUT a model. /api/predict will return 503.")
    yield
    reset_predictor()


app = FastAPI(
    title="IWNET Grape Leaf Disease Classification",
    description=(
        "EfficientNet-B3 + IW attention + adaptive scale fusion for grape leaf "
        "disease classification. Seven classes, one of which is a rejection "
        "class for images that are not grape leaves."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_config() -> Config:
    if _CONFIG is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return _CONFIG


def get_ready_predictor() -> Predictor:
    """Dependency that returns the predictor, or 503 with an actionable message."""
    try:
        return get_predictor(get_config())
    except (CheckpointError, InferenceError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model is available. Train it first with "
                "`python grape.py`, then restart the server."
            ),
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ClassProbability(BaseModel):
    name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    prediction: str
    label: str
    confidence: float
    is_rejection: bool
    description: str
    note: str
    probabilities: list[ClassProbability]
    timing_ms: dict[str, float]
    model_trained_at_utc: str | None = None
    dataset_fingerprint_sha256: str | None = None
    confidence_note: str
    #: Present only when the request asked for ``?visualize=true``. ``None``
    #: keeps the existing lightweight response shape identical.
    visualization: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    classes: int
    dataset: str | None = None
    detail: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _decode_image(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"That file could not be read as an image ({type(exc).__name__}).",
        ) from exc


def _to_response(prediction) -> PredictResponse:
    return PredictResponse(
        prediction=prediction.predicted_class,
        label=prediction.class_info.get("label", prediction.predicted_class),
        confidence=prediction.confidence,
        is_rejection=prediction.is_rejection,
        description=prediction.class_info.get("description", ""),
        note=prediction.class_info.get("note", ""),
        probabilities=[
            ClassProbability(name=name, probability=value)
            for name, value in prediction.probabilities.items()
        ],
        timing_ms={
            "preprocessing": round(prediction.preprocessing_ms, 2),
            "inference": round(prediction.inference_ms, 2),
        },
        model_trained_at_utc=prediction.model_trained_at,
        dataset_fingerprint_sha256=prediction.dataset_fingerprint,
        confidence_note=prediction.to_dict()["confidence_note"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe. Reports model state instead of failing when untrained."""
    cfg = _CONFIG
    try:
        predictor = get_predictor(cfg) if cfg is not None else None
    except (CheckpointError, InferenceError):
        predictor = None

    loaded = predictor is not None and predictor.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        device=str(predictor._device) if loaded else None,
        classes=len(FINAL_CLASSES),
        dataset=str(cfg.paths.dataset_root) if cfg is not None else None,
        detail=None if loaded else _STARTUP_ERROR,
    )


@app.get("/api/model", tags=["system"])
def model_info(predictor: Annotated[Predictor, Depends(get_ready_predictor)]) -> dict[str, Any]:
    meta = predictor.metadata
    # The checkpoint stores `val_acc` / `seed` / `*_version`. Asking for
    # `val_accuracy` / `dataset_seed` / `versions` returned nulls for a model
    # that had all of them, which reads as "nothing was recorded" rather than
    # as the bug it was.
    return {
        "architecture": "IWNET (EfficientNet-B3 + IW attention + adaptive scale fusion)",
        "backbone": meta.get("backbone"),
        "image_size": predictor.image_size,
        "num_classes": len(predictor.classes),
        "classes": predictor.classes,
        "classes_are_informational_only": True,
        "trained_at_utc": meta.get("trained_at_utc"),
        "epoch": meta.get("epoch"),
        "val_accuracy": meta.get("val_acc"),
        "val_loss": meta.get("val_loss"),
        "train_accuracy": meta.get("train_acc"),
        "dataset_fingerprint_sha256": meta.get("dataset_fingerprint_sha256"),
        "dataset_built_at_utc": meta.get("dataset_built_at_utc"),
        "dataset_seed": meta.get("seed"),
        "git_commit": meta.get("git_commit"),
        "versions": {
            "python": meta.get("python_version"),
            "torch": meta.get("torch_version"),
            "torchvision": meta.get("torchvision_version"),
            "timm": meta.get("timm_version"),
        },
        "load_seconds": round(predictor.load_seconds, 3),
        "device": str(predictor._device),
    }


@app.get("/api/classes", tags=["system"])
def classes() -> list[dict[str, Any]]:
    return class_order()


@app.post("/api/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    request: Request,
    predictor: Annotated[Predictor, Depends(get_ready_predictor)],
    file: Annotated[UploadFile | None, File(description="Image file")] = None,
    image_path: Annotated[str | None, Query(description="Server-side dataset path")] = None,
    visualize: Annotated[
        bool,
        Query(
            description=(
                "Attach an interpretability payload describing how the image was "
                "processed. Off by default: it adds response time and payload size. "
                "Never changes the prediction."
            )
        ),
    ] = False,
) -> PredictResponse:
    """Classify a grape leaf image.

    Accepts a multipart ``file`` upload, or a raw image body, or an
    ``image_path`` pointing at a server-side dataset image.

    Pass ``?visualize=true`` to additionally receive compact, display-only
    renderings of the preprocessing and attention stages. The prediction fields
    are identical either way, and a visualization failure never fails the
    prediction.
    """
    image: Image.Image | None = None
    include_original = True

    if file is not None:
        image = _decode_image(await file.read())
    elif image_path:
        resolved = _safe_dataset_path(image_path, predictor)
        # The caller addressed a server-side dataset image and has already shown
        # that thumbnail; echoing the bytes back would only bloat the response,
        # so the original-image stage reports itself unavailable instead.
        include_original = False
        try:
            with Image.open(resolved) as handle:
                image = handle.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read {image_path}") from exc
    else:
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            raise HTTPException(
                status_code=400,
                detail="Send multipart/form-data with a 'file' field, a raw image body, "
                       "or ?image_path=<dataset image>.",
            )
        image = _decode_image(await request.body())

    if image is None:
        raise HTTPException(status_code=400, detail="No image received.")

    if not visualize:
        try:
            return _to_response(predictor.predict_pil(image))
        except InferenceError as exc:
            log.warning("Prediction failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        prediction, trace, tensor = predictor.predict_pil_traced(image)
    except InferenceError as exc:
        log.warning("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = _to_response(prediction)
    try:
        from iwnet.inference.visualize import build_visualization

        response.visualization = build_visualization(
            image=image,
            transform=predictor._transform,
            normalized_tensor=tensor,
            trace=trace,
            prediction=prediction,
            include_original=include_original,
        )
    except Exception as exc:  # noqa: BLE001 - a visualization must never fail a prediction
        log.warning("Visualization unavailable: %s: %s", type(exc).__name__, exc)
        response.visualization = {
            "available": False,
            "message": "Some processing visualizations are unavailable.",
            "detail": type(exc).__name__,
        }
    finally:
        if trace is not None:
            trace.release()
    return response


def _safe_dataset_path(image_path: str, predictor: Predictor) -> str:
    """Resolve a path and refuse anything outside the dataset root."""
    cfg = get_config()
    root = os.path.realpath(str(cfg.paths.dataset_root))
    candidate = os.path.realpath(os.path.join(root, image_path))
    if not (candidate == root or candidate.startswith(root + os.sep)):
        raise HTTPException(
            status_code=403, detail="image_path must point inside the dataset directory."
        )
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail=f"No such dataset image: {image_path}")
    return candidate


@app.get("/api/samples", tags=["inference"])
def samples(
    predictor: Annotated[Predictor, Depends(get_ready_predictor)],
    per_class: Annotated[int, Query(ge=1, le=20)] = 2,
) -> list[dict[str, Any]]:
    """Labelled dataset images, for the UI's example gallery."""
    cfg = get_config()
    root = str(cfg.paths.dataset_root)
    grouped: dict[str, list[str]] = {}
    for name in predictor.classes:
        directory = os.path.join(root, "test", name)
        if not os.path.isdir(directory):
            continue
        files = sorted(f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        grouped[name] = files

    rng = random.Random(cfg.data.seed)
    out: list[dict[str, Any]] = []
    for name, files in grouped.items():
        for filename in rng.sample(files, k=min(per_class, len(files))):
            out.append(
                {
                    "class": name,
                    "filename": filename,
                    "url": f"/api/samples/{name}/{filename}",
                    "path": f"test/{name}/{filename}",
                    "note": "Ground truth from the held-out test split.",
                }
            )
    return out


@app.get("/api/samples/{class_name}/{filename}", tags=["inference"])
def sample_image(class_name: str, filename: str) -> FileResponse:
    """Serve one dataset image. Path traversal is blocked."""
    cfg = get_config()
    root = os.path.realpath(str(cfg.paths.dataset_root))
    candidate = os.path.realpath(os.path.join(root, "test", class_name, filename))
    if not candidate.startswith(root + os.sep):
        raise HTTPException(status_code=403, detail="Invalid path.")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(candidate)


@app.exception_handler(CheckpointError)
def _checkpoint_error_handler(request: Request, exc: CheckpointError) -> JSONResponse:
    log.error("Checkpoint error: %s", exc)
    return JSONResponse(status_code=503, detail=str(exc))


# Static UI last so it does not shadow /api routes.
if os.path.isdir(FRONTEND_DIR):  # pragma: no cover - presence depends on deployment
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
