"""Kiểm thử Backend: validate schema, 400 khi input sai."""

import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

os.environ.setdefault("AI_SERVICE_URL", "http://ai-service:8001")
os.environ.setdefault("MONGODB_URI", "mongodb://invalid:27017")

from app import app  # noqa: E402

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["service"] == "backend"
    assert "port" in body
    assert "uptime_sec" in body


def test_schema_matches_features():
    res = client.get("/api/schema")
    assert res.status_code == 200
    names = [f["name"] for f in res.json()["features"]]
    assert names == ["age", "sex", "bmi", "children", "smoker", "region"] or set(names) == {
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
    }


def test_predict_invalid_input():
    res = client.post("/api/predict", json={"features": {"age": 30}})
    assert res.status_code == 400
    body = res.json()
    assert body["error"] == "invalid_input"
    assert "request_id" in body


def test_predict_bad_smoker():
    payload = {
        "features": {
            "age": 30,
            "sex": "male",
            "bmi": 25.5,
            "children": 2,
            "smoker": "YES",
            "region": "southeast",
        }
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 400
    assert "smoker" in res.json()["detail"]


def test_history_empty_or_list():
    res = client.get("/api/history")
    assert res.status_code == 200
    assert "items" in res.json()
