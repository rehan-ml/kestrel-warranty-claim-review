# Submission form draft

## Public GitHub repository

`[ADD REPOSITORY URL]`

## Short screen recording

`[ADD PRIVATE/VIEW-ONLY VIDEO URL]`

## Honest time spent

`[ADD ACTUAL HOURS AFTER COMPLETION]`

## What I built

I built a warranty-claim decision-support tool that assigns a fraud-risk score to every claim, ranks a limited human-review queue, explains the main risk signals, and exposes the result through both a JSON endpoint and a simple reviewer screen. I used an honest time-based validation split and evaluated PR-AUC, queue precision/recall, and rupee impact instead of relying on misleading overall accuracy. The public repository contains reproducible code, tests, evidence, the required two-column predictions, and a non-technical memo; raw client data and the trained model remain excluded.

## Expected PR-AUC on the hidden/unlabelled period

Approximately **0.60**. The June 2026 temporal holdout achieved 0.650 PR-AUC, but I would expect some degradation because the unlabelled period is later and the May policy change created concept drift. This estimate is intentionally more conservative than the observed holdout result.

## Production model availability

Yes. `predictions.csv` is included in the submission repository as explicitly required. The trained model is available through a secure private channel because the supplied policy prohibits publishing raw client data and sensitive artifacts.

## AI tools and technologies used

Technologies: Python, pandas, scikit-learn, Flask, HTML/CSS/JavaScript, joblib, and pytest. OpenAI Codex was used as a development assistant for code scaffolding, debugging, test suggestions, and documentation review. I reviewed the modelling decisions, validation evidence, privacy handling, and final implementation. The submitted service itself does not require a paid AI API.
