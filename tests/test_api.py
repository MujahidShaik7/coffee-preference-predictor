"""
test_api.py
-----------
Basic tests for the Coffee Preference Predictor Flask API.

Run:
    python -m pytest
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app import app as flask_app


VALID_PAYLOAD = {
    "strength": "Medium",
    "milk": "High",
    "sweetness": 4,
    "temperature": "Hot",
    "bitterness": 2,
    "creaminess": 5,
    "caffeine": "Medium",
    "time_of_day": "Morning",
    "flavor": "Caramel",
    "experience": "Sweet & Creamy",
    "previous": "Latte",
}


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_homepage_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Coffee Preference Predictor" in resp.data or b"Cupwise" in resp.data


def test_valid_prediction(client):
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.get_json()

    assert "prediction" in data
    assert isinstance(data["prediction"], str)
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "description" in data
    assert "personality" in data
    assert "why_this_matches_you" in data
    assert isinstance(data["why_this_matches_you"], list)


def test_missing_field_returns_400(client):
    payload = dict(VALID_PAYLOAD)
    del payload["milk"]
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_invalid_category_returns_400(client):
    payload = dict(VALID_PAYLOAD)
    payload["strength"] = "Extreme"
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_invalid_numeric_range_returns_400(client):
    payload = dict(VALID_PAYLOAD)
    payload["sweetness"] = 10
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_non_json_body_returns_400(client):
    resp = client.post("/api/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_prediction_response_structure(client):
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    data = resp.get_json()
    expected_keys = {
        "prediction", "confidence", "description", "best_for", "tagline",
        "personality", "personality_description", "recommendation",
        "why_this_matches_you", "top_matches",
    }
    assert expected_keys.issubset(data.keys())
    assert len(data["top_matches"]) == 3
