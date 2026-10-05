"""
app.py
------
Flask backend for the Coffee Preference Predictor.

Loads the trained ML pipeline ONCE at startup and serves:
    GET  /              -> the web app (index.html)
    GET  /api/health     -> health check
    POST /api/predict    -> run a prediction against the trained model
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config.coffee_data import get_coffee_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "coffee_preference_model.pkl")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------
# Load model once at startup (not per-request)
# ---------------------------------------------------------------------
_model = None
_model_error = None
try:
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    else:
        _model_error = "Model file not found. Run `python ml/train_model.py` first."
except Exception as exc:  # noqa: BLE001
    _model_error = f"Failed to load model: {exc}"

# ---------------------------------------------------------------------
# Input validation schema
# ---------------------------------------------------------------------
CATEGORICAL_CHOICES = {
    "strength": {"Light", "Medium", "Strong"},
    "milk": {"None", "Low", "Medium", "High"},
    "temperature": {"Hot", "Iced", "Cold"},
    "caffeine": {"Low", "Medium", "High"},
    "time_of_day": {"Morning", "Afternoon", "Evening", "Night"},
    "flavor": {"Chocolate", "Nutty", "Fruity", "Caramel", "Classic", "Vanilla"},
    "experience": {"Quick & Strong", "Smooth & Relaxing", "Sweet & Creamy", "Refreshing & Cold"},
    "previous": {
        "Espresso", "Cappuccino", "Latte", "Americano",
        "Mocha", "Cold Brew", "Filter Coffee", "None",
    },
}
NUMERIC_FIELDS = {"sweetness": (1, 5), "bitterness": (1, 5), "creaminess": (1, 5)}

REQUIRED_FIELDS = list(CATEGORICAL_CHOICES.keys()) + list(NUMERIC_FIELDS.keys())


def validate_payload(payload: dict):
    """Return (cleaned_dict, error_message). error_message is None if valid."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, "")]
    if missing:
        return None, f"Missing required field(s): {', '.join(missing)}"

    cleaned = {}

    for field, choices in CATEGORICAL_CHOICES.items():
        value = str(payload[field]).strip()
        if value not in choices:
            return None, f"Invalid value for '{field}': {value!r}. Expected one of {sorted(choices)}."
        cleaned[field] = value

    for field, (lo, hi) in NUMERIC_FIELDS.items():
        raw = payload[field]
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None, f"Field '{field}' must be an integer between {lo} and {hi}."
        if not (lo <= value <= hi):
            return None, f"Field '{field}' must be between {lo} and {hi}."
        cleaned[field] = value

    return cleaned, None


def build_explanation(cleaned: dict, prediction: str):
    """Generate simple, honest 'why this matches you' bullet points based
    on the user's own answers (not fabricated feature-importance numbers)."""
    bullets = []

    strength_map = {
        "Strong": "You prefer a strong, intense coffee.",
        "Medium": "You prefer a moderately strong coffee.",
        "Light": "You prefer a lighter coffee.",
    }
    bullets.append(strength_map[cleaned["strength"]])

    if cleaned["milk"] in ("Medium", "High"):
        bullets.append("You enjoy a good amount of milk in your coffee.")
    elif cleaned["milk"] == "None":
        bullets.append("You prefer your coffee without milk.")

    if cleaned["creaminess"] >= 4:
        bullets.append("You lean toward creamy, smooth textures.")
    if cleaned["sweetness"] >= 4:
        bullets.append("You like sweeter flavors.")
    if cleaned["bitterness"] >= 4:
        bullets.append("You have a high tolerance for bitterness.")

    if cleaned["temperature"] in ("Iced", "Cold"):
        bullets.append("You prefer a chilled, refreshing coffee.")
    else:
        bullets.append(f"You usually reach for coffee in the {cleaned['time_of_day'].lower()}.")

    bullets.append(f"Your flavor preference leans {cleaned['flavor'].lower()}.")
    bullets.append(f"You're drawn to a '{cleaned['experience'].lower()}' coffee experience.")

    # keep it focused -- top 4 most relevant bullets
    return bullets[:4]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    if _model is None:
        return jsonify({"status": "error", "detail": _model_error}), 503
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({
            "error": "Prediction model is not available. Please try again later."
        }), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    cleaned, error = validate_payload(payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        feature_order = [
            "strength", "milk", "temperature", "caffeine",
            "time_of_day", "flavor", "experience", "previous",
            "sweetness", "bitterness", "creaminess",
        ]
        X = pd.DataFrame([{k: cleaned[k] for k in feature_order}])

        proba = _model.predict_proba(X)[0]
        classes = _model.classes_
        best_idx = int(np.argmax(proba))
        prediction = classes[best_idx]
        confidence = float(proba[best_idx])

        # top 3 matches for a nicer, honest result
        order = np.argsort(proba)[::-1][:3]
        top_matches = [
            {"coffee": classes[i], "confidence": round(float(proba[i]), 4)}
            for i in order
        ]

        info = get_coffee_info(prediction)
        explanation = build_explanation(cleaned, prediction)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "description": info["description"],
            "best_for": info["best_for"],
            "tagline": info["tagline"],
            "personality": info["personality_name"],
            "personality_description": info["personality_description"],
            "recommendation": info["description"],
            "why_this_matches_you": explanation,
            "top_matches": top_matches,
        })
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Something went wrong while finding your coffee match. Please try again."
        }), 500


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "An unexpected server error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
