"""
Frontend — phục vụ giao diện tĩnh và proxy /api/* tới Backend.
Ghi log có request_id mỗi khi người dùng bấm Dự đoán.
Địa chỉ Backend đọc từ BACKEND_INTERNAL_URL (Docker) hoặc API_URL.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
SERVICE_NAME = "frontend"
STARTED_AT = time.time()
PORT = int(os.environ.get("FRONTEND_PORT", "80"))
BACKEND_INTERNAL_URL = os.environ.get(
    "BACKEND_INTERNAL_URL",
    os.environ.get("API_URL", "http://backend:8000"),
).rstrip("/")
PUBLIC_DIR = Path(__file__).resolve().parent / "public"


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

app = FastAPI(title="Insurance Frontend", version="1.0.0")


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
    if request.url.path.startswith("/api") or request.url.path in ("/", "/health"):
        logger.info("%s %s -> %s in %.0fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "port": PORT,
        "uptime_sec": round(time.time() - STARTED_AT, 1),
        "backend_url": BACKEND_INTERNAL_URL,
    }


@app.get("/config.js")
def config_js():
    """Browser gọi /api trên cùng origin (FE proxy). Không hardcode localhost."""
    body = "window.APP_CONFIG = { apiBase: '' };\n"
    return Response(content=body, media_type="application/javascript")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_api(path: str, request: Request):
    rid = request.state.request_id
    if path == "predict" and request.method == "POST":
        logger.info("click Dự đoán -> POST /api/predict")

    url = f"{BACKEND_INTERNAL_URL}/api/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()
    headers = {
        "X-Request-ID": rid,
        "Content-Type": request.headers.get("content-type", "application/json"),
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.request(request.method, url, content=body, headers=headers)
    except httpx.HTTPError as exc:
        logger.error("không kết nối backend: %s", type(exc).__name__)
        return JSONResponse(
            status_code=502,
            content={"error": "backend_unavailable", "detail": str(exc), "request_id": rid},
        )

    logger.info("backend %s /api/%s -> %s", request.method, path, res.status_code)
    return Response(
        content=res.content,
        status_code=res.status_code,
        media_type=res.headers.get("content-type", "application/json"),
        headers={"X-Request-ID": rid},
    )


@app.get("/")
def index():
    return FileResponse(PUBLIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    logger.info("khởi động %s trên 0.0.0.0:%s backend=%s", SERVICE_NAME, PORT, BACKEND_INTERNAL_URL)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
