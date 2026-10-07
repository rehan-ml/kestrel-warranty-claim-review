from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "partner_age_days",
    "partner_is_new",
    "amount_to_price",
    "purchase_to_warranty",
    "outside_warranty",
    "post_may_policy",
    "small_claim",
    "post_may_small_claim",
    "photo_missing",
    "inspection_missing",
    "inspector_note_missing",
    "serial_format_error",
    "description_length",
    "note_length",
    "submit_month",
    "submit_weekday",
    "submit_hour",
]

CATEGORICAL_FEATURES = [
    "partner_id",
    "sku",
    "photo_attached",
    "partner_inspected",
    "claim_description",
    "inspector_note",
    "source",
    "city",
    "partner_type",
    "family",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalise_serial(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


def prepare_records(
    records: pd.DataFrame,
    partners: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    out = records.copy()
    out["submitted_at"] = pd.to_datetime(out["submitted_at"], errors="coerce")
    out["serial_norm"] = normalise_serial(out["product_serial"])
    raw_serial = out["product_serial"].fillna("").astype(str).str.upper()
    out["serial_format_error"] = (raw_serial != out["serial_norm"]).astype(int)

    out = out.merge(partners, on="partner_id", how="left", validate="many_to_one")
    out = out.merge(products, on="sku", how="left", validate="many_to_one")
    out["onboarded_date"] = pd.to_datetime(out["onboarded_date"], errors="coerce")
    out["partner_age_days"] = (
        out["submitted_at"] - out["onboarded_date"]
    ).dt.days.clip(lower=0)
    out["partner_is_new"] = (out["partner_age_days"] < 365).astype(int)
    out["amount_to_price"] = out["claim_amount_inr"] / out["list_price_inr"].replace(0, np.nan)
    out["warranty_days"] = out["warranty_months"] * 30.4375
    out["purchase_to_warranty"] = out["days_since_purchase"] / out["warranty_days"].replace(0, np.nan)
    out["outside_warranty"] = (out["days_since_purchase"] > out["warranty_days"]).astype(int)
    out["post_may_policy"] = (out["submitted_at"] >= pd.Timestamp("2026-05-01")).astype(int)
    out["small_claim"] = (out["claim_amount_inr"] < 2000).astype(int)
    out["post_may_small_claim"] = out["post_may_policy"] * out["small_claim"]
    out["photo_missing"] = (out["photo_attached"] != "Y").astype(int)
    out["inspection_missing"] = (out["partner_inspected"] != "Y").astype(int)
    out["inspector_note_missing"] = out["inspector_note"].isna().astype(int)
    out["description_length"] = out["claim_description"].fillna("").str.len()
    out["note_length"] = out["inspector_note"].fillna("").str.len()
    out["submit_month"] = out["submitted_at"].dt.month
    out["submit_weekday"] = out["submitted_at"].dt.weekday
    out["submit_hour"] = out["submitted_at"].dt.hour
    return out


def make_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def fit_regime_history(labelled: pd.DataFrame) -> dict[str, Any]:
    regime = labelled[
        (labelled["submitted_at"] >= pd.Timestamp("2026-05-01"))
        & (labelled["claim_amount_inr"] < 2000)
        & (labelled["partner_inspected"] != "Y")
    ]
    if regime.empty:
        return {"default_rate": float(labelled["target"].mean()), "partner_rates": {}, "partner_counts": {}}
    stats = regime.groupby("partner_id")["target"].agg(["count", "mean"])
    return {
        "default_rate": float(regime["target"].mean()),
        "partner_rates": stats["mean"].astype(float).to_dict(),
        "partner_counts": stats["count"].astype(int).to_dict(),
    }


def score_records(bundle: dict[str, Any], prepared: pd.DataFrame) -> np.ndarray:
    model_scores = bundle["pipeline"].predict_proba(prepared[MODEL_FEATURES])[:, 1]
    history = bundle["regime_history"]
    partner_rates = prepared["partner_id"].map(history["partner_rates"])
    regime_scores = partner_rates.fillna(history["default_rate"]).to_numpy(dtype=float)
    regime_mask = (
        (prepared["submitted_at"] >= pd.Timestamp("2026-05-01"))
        & (prepared["claim_amount_inr"] < 2000)
        & (prepared["partner_inspected"] != "Y")
    ).to_numpy()
    # The tiny term only breaks equal partner-history scores deterministically.
    scores = np.where(regime_mask, regime_scores + 1e-6 * model_scores, model_scores)
    return np.clip(scores, 0.0, 1.0)


def explain_record(bundle: dict[str, Any], row: pd.Series, score: float) -> list[str]:
    reasons: list[str] = []
    history = bundle["regime_history"]
    partner = str(row.get("partner_id", ""))
    regime_case = (
        row.get("submitted_at") >= pd.Timestamp("2026-05-01")
        and float(row.get("claim_amount_inr", 0)) < 2000
        and row.get("partner_inspected") != "Y"
    )
    if regime_case:
        reasons.append("Sub-Rs 2,000 claim without inspection after the 1 May auto-approval change")
        if partner in history["partner_rates"]:
            rate = 100 * history["partner_rates"][partner]
            count = history["partner_counts"].get(partner, 0)
            reasons.append(f"Partner's comparable post-policy history: {rate:.1f}% flagged across {count} claims")
    if float(row.get("amount_to_price", 0) or 0) >= 0.6:
        reasons.append("Claim amount is high relative to the product list price")
    if int(row.get("customer_prior_claims", 0) or 0) >= 2:
        reasons.append("Customer has multiple prior warranty claims")
    if row.get("photo_attached") != "Y":
        reasons.append("No supporting photo is attached")
    if int(row.get("serial_format_error", 0) or 0):
        reasons.append("Product serial required normalisation")
    if not reasons:
        reasons.append("Combined partner, product, claim and text-pattern signals")
    return reasons[:4]


def fallback_score(record: dict[str, Any]) -> tuple[float, list[str]]:
    submitted = pd.to_datetime(record.get("submitted_at"), errors="coerce")
    amount = float(record.get("claim_amount_inr", 0) or 0)
    inspected = record.get("partner_inspected", "N")
    photo = record.get("photo_attached", "N")
    prior = int(record.get("customer_prior_claims", 0) or 0)
    score = 0.02
    reasons: list[str] = []
    if pd.notna(submitted) and submitted >= pd.Timestamp("2026-05-01") and amount < 2000 and inspected != "Y":
        score += 0.20
        reasons.append("Post-policy sub-Rs 2,000 claim without inspection")
    if photo != "Y":
        score += 0.03
        reasons.append("No supporting photo")
    if prior >= 2:
        score += min(0.12, 0.04 * prior)
        reasons.append("Multiple previous claims")
    return min(score, 0.95), reasons or ["No strong rule-based warning found"]

