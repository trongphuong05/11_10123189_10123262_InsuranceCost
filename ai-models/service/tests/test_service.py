"""Kiểm thử AI Service: model nạp lúc import, predict hợp lệ/không hợp lệ."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app import MODEL, app  # noqa: E402

client = TestClient(app)

VALID = {
    "features": {
        "age": 30,
        "sex": "male",
        "bmi": 25.5,
        "children": 2,
        "smoker": "no",
        "region": "southeast",
    }
}


def test_model_loaded_at_import():
    assert MODEL is not None


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "port" in body
    assert "uptime_sec" in body


def test_model_info_has_metrics():
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["model_version"] == "1.0.0"
    assert "R2" in body["metrics"]
    assert body["schema"]["target"]["name"] == "charges"


def test_predict_ok():
    res = client.post("/predict", json=VALID, headers={"X-Request-ID": "testreq001"})
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["prediction"], float)
    assert body["prediction"] > 0
    assert body["model_version"] == "1.0.0"
    assert body["request_id"] == "testreq001"
    assert res.headers.get("x-request-id") == "testreq001"


def test_predict_missing_field():
    bad = {"features": {"age": 30, "sex": "male"}}
    res = client.post("/predict", json=bad)
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_input"


def test_predict_invalid_enum():
    payload = {**VALID, "features": {**VALID["features"], "smoker": "maybe"}}
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    assert "smoker" in res.json()["detail"]


def test_predict_out_of_range():
    payload = {**VALID, "features": {**VALID["features"], "age": 9}}
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
