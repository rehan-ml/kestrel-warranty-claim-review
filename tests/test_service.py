import importlib


def test_health_and_fallback_prediction(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "missing-model.joblib")
    service = importlib.import_module("src.service")
    client = service.app.test_client()
    assert client.get("/health").status_code == 200
    payload = {
        "submitted_at": "2026-07-04T10:30:00",
        "partner_id": "DEMO-PARTNER",
        "sku": "DEMO-SKU",
        "product_serial": "DEMO123456",
        "days_since_purchase": 90,
        "claim_amount_inr": 1499,
        "photo_attached": "N",
        "partner_inspected": "N",
        "claim_description": "unit not heating",
        "inspector_note": None,
        "customer_prior_claims": 1,
        "source": "crm",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["recommendation"] in {"manual_review", "approve_normally"}
    assert body["reasons"]


def test_missing_fields_are_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "missing-model.joblib")
    service = importlib.import_module("src.service")
    response = service.app.test_client().post("/api/predict", json={"partner_id": "SP1"})
    assert response.status_code == 400
    assert "fields" in response.get_json()

