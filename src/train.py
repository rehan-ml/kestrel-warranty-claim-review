from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .modeling import (
        MODEL_FEATURES,
        fit_regime_history,
        make_pipeline,
        prepare_records,
        score_records,
    )
except ImportError:  # Allows direct execution from the repository root.
    from modeling import (
        MODEL_FEATURES,
        fit_regime_history,
        make_pipeline,
        prepare_records,
        score_records,
    )


def load_pack(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_unlabelled.csv")
    partners = pd.read_csv(data_dir / "partners.csv")
    products = pd.read_csv(data_dir / "products.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, partners, products, sample


def economics(valid: pd.DataFrame, scores: np.ndarray, k: int) -> dict[str, float]:
    order = np.argsort(-scores)[:k]
    reviewed = valid.iloc[order]
    caught = int(reviewed["target"].sum())
    genuine = k - caught
    fraud_value = float(reviewed.loc[reviewed["target"] == 1, "claim_amount_inr"].sum())
    review_cost = 260.0 * k
    goodwill_cost = 380.0 * genuine
    pred = np.zeros(len(valid), dtype=int)
    pred[order] = 1
    return {
        "reviewed": k,
        "frauds_caught": caught,
        "precision": caught / k,
        "recall": caught / int(valid["target"].sum()),
        "accuracy": float((pred == valid["target"].to_numpy()).mean()),
        "fraud_value_caught_inr": fraud_value,
        "review_cost_inr": review_cost,
        "goodwill_cost_inr": goodwill_cost,
        "net_value_inr": fraud_value - review_cost - goodwill_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    train_raw, test_raw, partners, products, sample = load_pack(args.data_dir)
    prepared = prepare_records(train_raw, partners, products)
    prepared = prepared.sort_values("submitted_at").drop_duplicates("claim_id", keep="first")
    labelled = prepared[prepared["is_fraud"].notna()].copy()
    labelled["target"] = labelled["is_fraud"].astype(int)

    # Honest temporal holdout: train through May, validate on June, which is the
    # latest labelled month and matches the post-policy operating regime.
    dev = labelled[labelled["submitted_at"] < pd.Timestamp("2026-06-01")].copy()
    valid = labelled[
        (labelled["submitted_at"] >= pd.Timestamp("2026-06-01"))
        & (labelled["submitted_at"] < pd.Timestamp("2026-07-01"))
    ].copy()
    validation_bundle = {
        "pipeline": make_pipeline(),
        "regime_history": fit_regime_history(dev),
    }
    validation_bundle["pipeline"].fit(dev[MODEL_FEATURES], dev["target"])
    valid_scores = score_records(validation_bundle, valid)
    order_20 = np.argsort(-valid_scores)[:20]
    threshold = float(valid_scores[order_20[-1]])
    metrics = {
        "split": "June 2026 temporal holdout",
        "validation_rows": int(len(valid)),
        "validation_frauds": int(valid["target"].sum()),
        "overall_labelled_rows": int(len(labelled)),
        "overall_fraud_rate": float(labelled["target"].mean()),
        "pr_auc": float(average_precision_score(valid["target"], valid_scores)),
        "roc_auc": float(roc_auc_score(valid["target"], valid_scores)),
        "review_20": economics(valid, valid_scores, 20),
        "review_40": economics(valid, valid_scores, 40),
        "review_threshold": threshold,
    }

    # Refit on every resolved claim for the future unlabelled period.
    final_bundle = {
        "pipeline": make_pipeline(),
        "regime_history": fit_regime_history(labelled),
        "review_threshold": threshold,
        "partners": partners,
        "products": products,
        "metrics": metrics,
        "version": "kestrel-warranty-v1",
    }
    final_bundle["pipeline"].fit(labelled[MODEL_FEATURES], labelled["target"])
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_bundle, args.artifact)

    test_prepared = prepare_records(test_raw, partners, products)
    test_scores = score_records(final_bundle, test_prepared)
    predictions = sample[["claim_id"]].copy()
    score_map = dict(zip(test_prepared["claim_id"], test_scores))
    predictions["score"] = predictions["claim_id"].map(score_map)
    if predictions["score"].isna().any():
        raise ValueError("Missing scores for one or more sample-submission claim IDs")
    if len(predictions) != len(test_raw) or not predictions["claim_id"].is_unique:
        raise ValueError("Prediction shape or claim IDs do not match the task contract")
    predictions.to_csv(args.predictions, index=False)
    (args.artifact.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.artifact}")
    print(f"Wrote {args.predictions}")


if __name__ == "__main__":
    main()

