"""
Backend — validate theo schema.json, gọi AI Service, lưu lịch sử MongoDB.
Mọi địa chỉ (AI_SERVICE_URL, MONGODB_URI) đọc từ biến môi trường.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
SERVICE_NAME = "backend"
STARTED_AT = time.time()
PORT = int(os.environ.get("BACKEND_PORT", "8000"))
AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://ai-service:8001").rstrip("/")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongodb:27017/insurance")
MONGODB_DB = os.environ.get("MONGODB_DB", "insurance")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", os.path.join(os.path.dirname(__file__), "schema.json"))


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


def load_schema(primary_path: str) -> dict[str, Any]:
    """Thử tải schema từ đường dẫn biến môi trường, nếu không tìm thấy sẽ thử các vị trí dự phòng."""
    possible_paths = [
        primary_path,
        os.path.join(os.path.dirname(__file__), "schema.json"),
        os.path.join(os.path.dirname(__file__), "models", "schema.json"),
        "/app/schema.json",
        "/app/models/schema.json",
    ]

    # Loại bỏ các đường dẫn trùng lặp nhưng giữ nguyên thứ tự ưu tiên
    seen = set()
    unique_paths = [p for p in possible_paths if p and not (p in seen or seen.add(p))]

    for path in unique_paths:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("Đã tải thành công schema từ: %s", path)
                    return data
            except Exception as err:
                logger.warning("Không thể đọc file schema tại %s: %s", path, err)

    logger.error("CẢNH BÁO: Không tìm thấy file schema.json ở bất kỳ vị trí nào! Sử dụng schema rỗng.")
    return {"features": []}


SCHEMA: dict[str, Any] = load_schema(SCHEMA_PATH)

_mongo_client: MongoClient | None = None
_history: Collection | None = None
_memory_history: list[dict] = []


def get_history_collection() -> Collection | None:
    global _mongo_client, _history
    if _history is not None:
        return _history
    try:
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        _mongo_client.admin.command("ping")
        _history = _mongo_client[MONGODB_DB]["predictions"]
        logger.info("kết nối MongoDB OK uri_host_ẩn")
        return _history
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB chưa sẵn sàng (%s) — dùng bộ nhớ tạm", type(exc).__name__)
        return None


def save_history(doc: dict) -> None:
    col = get_history_collection()
    stored = {**doc, "_id": doc["request_id"]}
    if col is None:
        _memory_history.insert(0, stored)
        del _memory_history[200:]
        return
    try:
        col.insert_one(stored)
    except PyMongoError as exc:
        logger.warning("không lưu được MongoDB: %s — fallback RAM", type(exc).__name__)
        _memory_history.insert(0, stored)


def list_history(limit: int = 20) -> list[dict]:
    col = get_history_collection()
    if col is None:
        return _memory_history[:limit]
    try:
        cursor = col.find().sort("created_at", -1).limit(limit)
        rows = []
        for d in cursor:
            d.pop("_id", None)
            rows.append(d)
        return rows
    except PyMongoError:
        return _memory_history[:limit]


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


app = FastAPI(
    title="Insurance Backend",
    version="1.0.0",
    description="Validate schema → gọi AI Service → lưu lịch sử.",
)

origins = ["*"] if CORS_ORIGINS.strip() == "*" else [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    logger.info("%s %s -> %s in %.0fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health():
    mongo_ok = get_history_collection() is not None
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "port": PORT,
        "uptime_sec": round(time.time() - STARTED_AT, 1),
        "ai_service_url": AI_SERVICE_URL,
        "mongodb": "ok" if mongo_ok else "fallback_memory",
    }


@app.get("/api/schema")
def api_schema():
    return SCHEMA


@app.get("/api/model-info")
async def api_model_info(request: Request):
    rid = request.state.request_id
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"{AI_SERVICE_URL}/model-info",
                headers={"X-Request-ID": rid},
            )
        res.raise_for_status()
        return res.json()
    except httpx.HTTPError as exc:
        logger.error("không lấy được model-info: %s", type(exc).__name__)
        return JSONResponse(
            status_code=502,
            content={"error": "ai_unavailable", "detail": str(exc), "request_id": rid},
        )


@app.get("/api/history")
def api_history(limit: int = 20):
    rows = list_history(max(1, min(limit, 100)))
    return {"items": rows, "count": len(rows)}


@app.post("/api/predict")
async def api_predict(request: Request):
    t0 = time.perf_counter()
    rid = request.state.request_id

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_input", "detail": "Body không phải JSON", "request_id": rid},
        )

    features = body.get("features") if isinstance(body, dict) else None
    error = validate_features(features)
    if error:
        logger.info("validate FAIL: %s", error)
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_input", "detail": error, "request_id": rid},
        )

    logger.info("validate OK -> gọi ai-service")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                f"{AI_SERVICE_URL}/predict",
                json={"features": features},
                headers={"X-Request-ID": rid, "Content-Type": "application/json"},
            )
    except httpx.HTTPError as exc:
        logger.error("không kết nối được AI Service: %s", type(exc).__name__)
        return JSONResponse(
            status_code=502,
            content={
                "error": "ai_unavailable",
                "detail": "Không kết nối được AI Service",
                "request_id": rid,
            },
        )

    if res.status_code >= 400:
        detail = res.text
        try:
            detail = res.json()
        except Exception:
            pass
        logger.error("AI Service trả %s", res.status_code)
        code = res.status_code if res.status_code in (400, 422) else 502
        if isinstance(detail, dict):
            return JSONResponse(status_code=code, content={**detail, "request_id": rid})
        return JSONResponse(
            status_code=code,
            content={"error": "ai_error", "detail": detail, "request_id": rid},
        )

    result = res.json()
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    result["request_id"] = rid
    result["server_latency_ms"] = total_ms

    doc = {
        "request_id": rid,
        "features": features,
        "prediction": result.get("prediction"),
        "model_version": result.get("model_version"),
        "model_name": result.get("model_name"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": total_ms,
    }
    save_history(doc)
    logger.info(
        "200 OK total %.0fms, saved history prediction=%.2f model=%s",
        total_ms,
        float(result.get("prediction") or 0),
        result.get("model_version"),
    )
    return result


if __name__ == "__main__":
    import uvicorn

    logger.info("khởi động %s trên 0.0.0.0:%s AI_SERVICE_URL=%s", SERVICE_NAME, PORT, AI_SERVICE_URL)
    get_history_collection()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")