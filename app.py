import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger
import uvicorn

# ── Path setup
BASE_DIR   = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_DIR  = BASE_DIR / 'model'
HEATMAP_DIR = BASE_DIR / 'heatmaps'
LOG_DIR    = BASE_DIR / 'logs'
SAVED_MODEL_DIR = BASE_DIR / 'saved_model'

from model.predict import predict as run_predict
from model.model_loader import load_model, unload_model

# ── Logging
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "app.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# ── Config
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
ALLOWED_TYPES    = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MODEL_PATH       = os.getenv("MODEL_PATH", str(SAVED_MODEL_DIR / "detector_best.pt"))
BASE_URL         = os.getenv("BASE_URL", "http://localhost:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fake Image Detector API...")
    HEATMAP_DIR.mkdir(exist_ok=True)
    try:
        model_path = MODEL_PATH if Path(MODEL_PATH).exists() else None
        load_model(model_path)
        logger.info("Model loaded and warmed up.")
    except Exception as e:
        logger.error(f"Model load failed (dev mode): {e}")
    yield
    logger.info("Shutting down...")
    unload_model()


app = FastAPI(
    title="Fake vs Real Image Detector",
    description=(
        "Detect AI-generated images using a Hybrid CNN + Vision Transformer model "
        "with frequency-domain (FFT/DCT) analysis and GradCAM explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")


class PredictionResponse(BaseModel):
    prediction:      str
    confidence:      float
    probabilities:   dict
    artifact_score:  float
    heatmap_url:     Optional[str]
    processing_time: float
    image_info:      dict


class HealthResponse(BaseModel):
    status:    str
    model:     str
    timestamp: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - t0, 3)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed}s)")
    return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        model="HybridDetector v1.0",
        timestamp=time.time(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, BMP)"),
    generate_heatmap: bool = Query(True, description="Generate GradCAM heatmap"),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {ALLOWED_TYPES}",
        )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    logger.info(f"Received image: {file.filename} ({size_mb:.2f} MB)")

    try:
        result = run_predict(
            image_bytes,
            model_path=MODEL_PATH if Path(MODEL_PATH).exists() else None,
            generate_heatmap=generate_heatmap,
            base_url=BASE_URL,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Image processing error: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal inference error.")

    logger.info(
        f"Result: {result['prediction']} ({result['confidence']:.2%}) "
        f"| artifact={result['artifact_score']:.3f} | time={result['processing_time']}s"
    )
    return PredictionResponse(**result)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
