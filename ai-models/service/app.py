"""
AI Service — nạp pipeline+model ngay khi container khởi động.
POST /predict, GET /health, GET /model-info, GET /schema
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Structured logging + request_id xuyên suốt
# ---------------------------------------------------------------------------
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
SERVICE_NAME = "ai-service"
STARTED_AT = time.time()
PORT = int(os.environ.get("AI_SERVICE_PORT", "8001"))


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")[:12]
        record.service = SERVICE_NAME
        return True


def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s %(service)-11s req=%(request_id)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())
    logger = logging.getLogger(SERVICE_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging()

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(SERVICE_DIR)
DEFAULT_MODELS = os.path.join(AI_ROOT, "models")


def _path(env_key: str, filename: str) -> str:
    override = os.environ.get(env_key)
    if override:
        return override
    return os.path.join(DEFAULT_MODELS, filename)


MODEL_PATH = _path("MODEL_PATH", "model.joblib")
SCHEMA_PATH = _path("SCHEMA_PATH", "schema.json")
METADATA_PATH = _path("METADATA_PATH", "metadata.json")

MODEL = None
SCHEMA: dict[str, Any] = {}
METADATA: dict[str, Any] = {}


def load_artifacts() -> None:
    """Nạp model + schema + metadata ngay khi process start."""
    global MODEL, SCHEMA, METADATA
    logger.info("nạp model từ %s", MODEL_PATH)
    t0 = time.perf_counter()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        SCHEMA = json.load(f)
    with open(METADATA_PATH, encoding="utf-8") as f:
        METADATA = json.load(f)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "đã nạp pipeline+model version=%s class=%s in %.0fms",
        METADATA.get("model_version"),
        METADATA.get("model_class"),
        elapsed_ms,
    )


# Nạp NGAY khi module được import (uvicorn/python app.py) — không lazy.
load_artifacts()

app = FastAPI(
    title="Insurance AI Service",
    version=METADATA.get("model_version", "1.0.0"),
    description="Dự đoán chi phí bảo hiểm. Pipeline sklearn được nạp lúc khởi động.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    logger.info(
        "%s %s -> %s in %.0fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def validate_features(features: Any) -> str | None:
    if not isinstance(features, dict):
        return "Body phải có object 'features'"
    names = [f["name"] for f in SCHEMA["features"]]
    missing = [n for n in names if n not in features]
    if missing:
        return f"Thiếu trường bắt buộc: {missing}"
    extra = [k for k in features.keys() if k not in names]
    if extra:
        return f"Trường không nằm trong schema: {extra}"
    for spec in SCHEMA["features"]:
        name = spec["name"]
        value = features[name]
        if spec["type"] == "number":
            try:
                num = float(value)
            except (TypeError, ValueError):
                return f"{name} phải là số"
            if spec.get("min") is not None and num < spec["min"]:
                return f"{name} phải ≥ {spec['min']}"
            if spec.get("max") is not None and num > spec["max"]:
                return f"{name} phải ≤ {spec['max']}"
        elif spec["type"] == "categorical":
            allowed = spec.get("enum") or []
            if str(value) not in allowed:
                return f"{name} phải thuộc {set(allowed)}"
    return None


def features_to_frame(features: dict) -> pd.DataFrame:
    order = SCHEMA.get("feature_order") or [f["name"] for f in SCHEMA["features"]]
    row = {}
    for name in order:
        spec = next(s for s in SCHEMA["features"] if s["name"] == name)
        value = features[name]
        row[name] = [float(value) if spec["type"] == "number" else value]
    return pd.DataFrame(row)


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "error",
        "service": SERVICE_NAME,
        "port": PORT,
        "uptime_sec": round(time.time() - STARTED_AT, 1),
        "model_loaded": MODEL is not None,
        "model_version": METADATA.get("model_version"),
        "model_path": MODEL_PATH,
    }


@app.get("/schema")
def get_schema():
    return SCHEMA


@app.get("/model-info")
def model_info():
    return {
        "model_name": METADATA.get("model_display_name"),
        "model_key": METADATA.get("model_name"),
        "model_class": METADATA.get("model_class"),
        "model_version": METADATA.get("model_version"),
        "task": METADATA.get("task"),
        "target": METADATA.get("target"),
        "metrics": METADATA.get("metrics"),
        "metrics_train": METADATA.get("metrics_train"),
        "comparison": METADATA.get("comparison"),
        "chosen_reason": METADATA.get("chosen_reason"),
        "library_versions": METADATA.get("library_versions"),
        "trained_at": METADATA.get("trained_at"),
        "artifact": METADATA.get("artifact"),
        "predict_ms": METADATA.get("predict_ms"),
        "schema": SCHEMA,
    }


@app.post("/predict")
async def predict(request: Request):
    t0 = time.perf_counter()
    rid = getattr(request.state, "request_id", request_id_var.get("-"))

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_input",
                "detail": "Body không phải JSON",
                "request_id": rid,
            },
        )

    features = body.get("features", body if isinstance(body, dict) else None)
    error = validate_features(features)
    if error:
        logger.info("validate FAIL: %s", error)
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_input", "detail": error, "request_id": rid},
        )

    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model_not_loaded", "detail": "Model chưa nạp", "request_id": rid},
        )

    X = features_to_frame(features)
    value = float(MODEL.predict(X)[0])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    version = METADATA.get("model_version", "1.0.0")
    name = METADATA.get("model_display_name", "model")

    logger.info("predict %.2f USD model=%s p=- in %.0fms", value, version, elapsed_ms)

    return {
        "prediction": round(value, 2),
        "unit": SCHEMA.get("target", {}).get("unit", "USD"),
        "probability": None,
        "model_version": version,
        "model_name": name,
        "request_id": rid,
        "latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("khởi động %s trên 0.0.0.0:%s (model đã nạp)", SERVICE_NAME, PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
