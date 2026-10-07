from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    from .modeling import explain_record, fallback_score, prepare_records, score_records
except ImportError:  # Allows `python src/service.py` as well as package imports in tests.
    from modeling import explain_record, fallback_score, prepare_records, score_records


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def score_one(payload: dict):
    if bundle is None:
        score, reasons = fallback_score(payload)
        threshold = 0.20
        mode = "demo_fallback"
    else:
        prepared = prepare_records(pd.DataFrame([payload]), bundle["partners"], bundle["products"])
        score = float(score_records(bundle, prepared)[0])
        reasons = explain_record(bundle, prepared.iloc[0], score)
        threshold = float(bundle["review_threshold"])
        mode = "trained_model"
    return {
        "fraud_score": round(score, 6),
        "recommendation": "manual_review" if score >= threshold else "approve_normally",
        "review_threshold": round(threshold, 6),
        "reasons": reasons,
        "mode": mode,
        "warning": "Decision support only; never auto-reject a warranty claim.",
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "mode": "trained_model" if bundle else "demo_fallback"})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected one claim as a JSON object"}), 400
    required = [
        "submitted_at",
        "partner_id",
        "sku",
        "product_serial",
        "days_since_purchase",
        "claim_amount_inr",
        "photo_attached",
        "partner_inspected",
        "claim_description",
        "customer_prior_claims",
        "source",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        return jsonify({"error": "Missing required fields", "fields": missing}), 400
    try:
        return jsonify(score_one(payload))
    except Exception as exc:
        return jsonify({"error": f"Could not score claim: {exc}"}), 422


@app.get("/")
def index():
    return render_template("index.html", model_mode="trained model" if bundle else "demo fallback")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8520")), debug=False)

