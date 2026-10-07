# Kestrel Warranty Claim Review

A decision-support prototype that scores warranty claims for possible fraud and sends only the highest-risk claims to a human review queue. It includes a reproducible training pipeline, a JSON endpoint, and a simple reviewer screen. It does **not** auto-reject claims.

## Why this approach

The data is highly imbalanced, so ordinary accuracy is misleading: predicting every claim as genuine already exceeds 97% accuracy. The solution therefore uses:

- an honest time-based validation split rather than a random split;
- PR-AUC and review-queue precision/recall as the main quality measures;
- an operational and financial check for review queues of 20 and 40 claims;
- human review for uncertain/high-risk cases, with reasons shown to the reviewer;
- explicit handling of the 1 May 2026 auto-approval policy change.

## Repository contents

- `src/train.py` — trains, evaluates, saves the local model, and creates `predictions.csv`.
- `src/predict.py` — generates predictions with an existing local model.
- `src/service.py` — Flask web screen and `POST /api/predict` endpoint.
- `src/modeling.py` — feature preparation, model, scoring, and explanations.
- `EVIDENCE.md` — validation design, results, failures, and limitations.
- `memo_to_ritu.md` — one-page non-technical recommendation.
- `tests/` — endpoint checks.

## Privacy note

Kestrel's supplied CSV/PDF files, trained model, and detailed local metrics artifact are intentionally excluded from this public repository. The required two-column `predictions.csv` is included because it is an explicit scored deliverable. The public `data/README.md` explains the expected local filenames without exposing the raw client records.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --data-dir data
python src/service.py
```

Open `http://127.0.0.1:8520`.

The interface also starts without the trained artifact in a transparent `demo_fallback` mode, so a reviewer can inspect the endpoint and screen without receiving client data. Training and production scoring require the private data pack.

## JSON endpoint

```bash
curl -X POST http://127.0.0.1:8520/api/predict \
  -H "Content-Type: application/json" \
  -d '{"submitted_at":"2026-07-04T10:30:00","partner_id":"DEMO-PARTNER","sku":"DEMO-SKU","product_serial":"DEMO123456","days_since_purchase":90,"claim_amount_inr":1499,"photo_attached":"N","partner_inspected":"N","claim_description":"unit not heating","inspector_note":null,"customer_prior_claims":1,"source":"crm"}'
```

## Generate the required predictions

Place the supplied private files under `data/`, then run:

```bash
python src/train.py --data-dir data --predictions predictions.csv
```

The script preserves the exact claim order from `sample_submission.csv` and checks that every claim has one score between 0 and 1.

## Tests

```bash
python -m pytest -q
```

## AI and technology use

The solution uses Python, pandas, scikit-learn, Flask, HTML/CSS/JavaScript, joblib, and pytest. OpenAI Codex was used as a development assistant for code scaffolding, debugging, test suggestions, and documentation review. The modelling choices, checks, and final submission were reviewed by the candidate. No paid AI API is required to run the product.
