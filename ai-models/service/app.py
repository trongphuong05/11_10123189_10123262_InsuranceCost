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

MODEL: Any = None
SCHEMA: dict[str, Any] = {}
METADATA: dict[str, Any] = {}
LOAD_ERROR: str | None = None


def load_artifacts() -> None:
    """Nạp model + schema + metadata. Tránh crash process nếu file không tồn tại."""
    global MODEL, SCHEMA, METADATA, LOAD_ERROR
    logger.info("Bắt đầu nạp model từ %s", MODEL_PATH)
    t0 = time.perf_counter()

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Không tìm thấy file model: {MODEL_PATH}")
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(f"Không tìm thấy file schema: {SCHEMA_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Không tìm thấy file metadata: {METADATA_PATH}")

        MODEL = joblib.load(MODEL_PATH)
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            SCHEMA = json.load(f)
        with open(METADATA_PATH, encoding="utf-8") as f:
            METADATA = json.load(f)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        LOAD_ERROR = None
        logger.info(
            "Đã nạp pipeline+model version=%s class=%s trong %.0fms",
            METADATA.get("model_version"),
            METADATA.get("model_class"),
            elapsed_ms,
        )
    except Exception as e:
        LOAD_ERROR = str(e)
        logger.error("Lỗi khởi tạo model/schema/metadata: %s", str(e))


# Thử nạp ngay khi khởi chạy app
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
    # Lấy x-request-id (không phân biệt hoa/thường) hoặc tạo UUID mới
    rid = (
        request.headers.get("x-request-id")
        or request.headers.get("X-Request-ID")
        or uuid.uuid4().hex
    )
    token = request_id_var.set(rid)
    request.state.request_id = rid
    t0 = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error("Unhandled Server Error: %s", str(exc), exc_info=True)
        response = JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Lỗi hệ thống nội bộ tại AI Service",
                "detail": str(exc),
                "request_id": rid,
            },
        )
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


def validate_features(features: Any) -> tuple[str, dict[str, Any]] | None:
    """Kiểm tra tính hợp lệ của features. 
    Trả về Tuple (Thông báo lỗi chi tiết, Dict thông tin chi tiết cho BE) nếu có lỗi.
    """
    if not isinstance(features, dict):
        return "Body payload phải chứa object 'features'", {
            "expected_type": "object",
            "received_type": type(features).__name__,
        }

    schema_features = SCHEMA.get("features", [])
    names = [f["name"] for f in schema_features]

    # 1. Kiểm tra thiếu trường
    missing = [n for n in names if n not in features]
    if missing:
        return f"Thiếu các trường bắt buộc: {missing}", {"missing_fields": missing}

    # 2. Kiểm tra thừa trường
    extra = [k for k in features.keys() if k not in names]
    if extra:
        return f"Phát hiện các trường không thuộc schema: {extra}", {"extra_fields": extra}

    # 3. Validate kiểu dữ liệu và ràng buộc từng trường
    invalid_fields = []
    for spec in schema_features:
        name = spec["name"]
        value = features[name]
        spec_type = spec.get("type")

        if spec_type == "number":
            try:
                num = float(value)
                if spec.get("min") is not None and num < spec["min"]:
                    invalid_fields.append({
                        "field": name,
                        "error": f"Giá trị {num} nhỏ hơn mức tối thiểu cho phép ({spec['min']})",
                        "min": spec["min"],
                        "value": value,
                    })
                if spec.get("max") is not None and num > spec["max"]:
                    invalid_fields.append({
                        "field": name,
                        "error": f"Giá trị {num} lớn hơn mức tối đa cho phép ({spec['max']})",
                        "max": spec["max"],
                        "value": value,
                    })
            except (TypeError, ValueError):
                invalid_fields.append({
                    "field": name,
                    "error": f"Giá trị '{value}' không phải là kiểu số (number)",
                    "value": value,
                })

        elif spec_type == "categorical":
            allowed = spec.get("enum") or []
            if str(value) not in allowed:
                invalid_fields.append({
                    "field": name,
                    "error": f"Giá trị '{value}' không nằm trong danh sách cho phép",
                    "allowed_values": allowed,
                    "value": value,
                })

    if invalid_fields:
        messages = [item["error"] for item in invalid_fields]
        return f"Lỗi định dạng dữ liệu: {'; '.join(messages)}", {
            "invalid_fields": invalid_fields
        }

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
    """Health check endpoint: Trả về HTTP 503 nếu model chưa nạp/thiếu file."""
    is_healthy = MODEL is not None and LOAD_ERROR is None
    status_code = 200 if is_healthy else 503

    content = {
        "status": "ok" if is_healthy else "unhealthy",
        "service": SERVICE_NAME,
        "port": PORT,
        "uptime_sec": round(time.time() - STARTED_AT, 1),
        "model_loaded": is_healthy,
        "model_version": METADATA.get("model_version"),
        "model_path": MODEL_PATH,
    }
    if not is_healthy:
        content["error"] = LOAD_ERROR or "Model chưa được nạp vào bộ nhớ"

    return JSONResponse(status_code=status_code, content=content)


@app.get("/schema")
def get_schema():
    if not SCHEMA:
        return JSONResponse(
            status_code=503,
            content={"error": "SCHEMA_NOT_LOADED", "message": "Schema chưa được nạp"},
        )
    return SCHEMA


@app.get("/model-info")
def model_info():
    if not METADATA:
        return JSONResponse(
            status_code=503,
            content={"error": "METADATA_NOT_LOADED", "message": "Metadata chưa được nạp"},
        )
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

    # 1. Báo lỗi nếu Model chưa được nạp
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "MODEL_NOT_READY",
                "message": "Model AI chưa sẵn sàng hoặc nạp thất bại",
                "detail": LOAD_ERROR or "Model file không khả dụng",
                "request_id": rid,
            },
        )

    # 2. Parse Body JSON
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_JSON",
                "message": "Body payload không đúng định dạng JSON chuẩn",
                "request_id": rid,
            },
        )

    features = body.get("features", body if isinstance(body, dict) else None)

    # 3. Validate features kỹ lưỡng cho Backend
    val_result = validate_features(features)
    if val_result:
        error_msg, error_details = val_result
        logger.info("Validate FAIL: %s", error_msg)
        return JSONResponse(
            status_code=400,
            content={
                "error": "VALIDATION_ERROR",
                "message": error_msg,
                "details": error_details,
                "request_id": rid,
            },
        )

    # 4. Dự đoán
    try:
        X = features_to_frame(features)
        value = float(MODEL.predict(X)[0])
    except Exception as e:
        logger.error("Lỗi khi suy luận model (predict): %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "PREDICTION_FAILED",
                "message": "Có lỗi phát sinh trong quá trình suy luận mô hình",
                "detail": str(e),
                "request_id": rid,
            },
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    version = METADATA.get("model_version", "1.0.0")
    name = METADATA.get("model_display_name", "model")

    logger.info("predict %.2f USD model=%s in %.0fms", value, version, elapsed_ms)

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

    logger.info("Khởi động %s trên 0.0.0.0:%s", SERVICE_NAME, PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")