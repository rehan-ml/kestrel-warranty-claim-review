# Validation evidence

## Decision to support

The model does not decide whether a customer is fraudulent. It ranks incoming claims so a trained employee can inspect a small, affordable queue. Claims outside that queue continue through the normal warranty process.

## Data checks and planted defects found

1. **Accuracy trap:** fraud is rare. An all-genuine prediction already exceeds the requested 97% accuracy, while stopping no fraud.
2. **Duplicate claim numbers:** the training file contains resubmitted claim IDs. They were deduplicated before splitting so the same claim cannot appear in both development and validation data.
3. **Unresolved-label contamination:** unresolved CRM claims are blank, while the old Zoho export used `0` because it could not store blank labels. Blank outcomes were excluded, and legacy `0` labels remain a stated limitation.
4. **Policy change:** from 1 May 2026, claims below Rs 2,000 were auto-approved without inspection. This changed both behaviour and the available evidence, so a random split would overstate performance.
5. **Serial formatting:** serials were normalised for case, spaces, and punctuation instead of treating formatting differences as different products.
6. **Partner fairness:** new partners are not automatically treated as fraudulent. Partner history is used only for comparable post-policy claims and every flag remains subject to human review.
7. **Capacity is a maximum:** the desk can review up to 40 claims per month; that does not mean using all 40 is financially sensible.
8. **Privacy conflict:** the assignment requests a public repository, but the supplied policy forbids publishing client data. Raw inputs and trained artifacts are excluded from Git. The required two-column prediction deliverable is included because the hiring team explicitly requires it in the repository.

## Validation design

- Resolved claims before June 2026: development set.
- Resolved claims submitted in June 2026: untouched temporal validation set.
- Primary metric: PR-AUC, because it measures ranking quality for the rare positive class.
- Operational checks: precision, recall, fraud value caught, review cost, customer-goodwill cost, and net value at queue sizes 20 and 40.
- Model: a class-balanced logistic regression over claim, partner, product, text, and timing features, combined with post-policy partner history for comparable sub-Rs 2,000 uninspected claims.

This is more honest than random cross-validation because June is later than training and follows the same May policy change that governs the unlabelled period.

## Results on the June holdout

| Measure | Result |
|---|---:|
| Validation rows | 713 |
| Fraud cases | 22 |
| PR-AUC | 0.650 |
| ROC-AUC | 0.933 |
| Top-20 frauds caught | 13 of 22 |
| Top-20 precision | 65.0% |
| Top-20 recall | 59.1% |
| Top-20 fraud value identified | Rs 19,099 |
| Top-20 net value after supplied costs | Rs 11,239 |
| Top-40 frauds caught | 16 of 22 |
| Top-40 net value after supplied costs | Rs 2,359 |

Net value uses only the supplied assumptions: Rs 260 per reviewed claim, Rs 380 goodwill cost for each genuine claim held, and the full claim amount for a fraudulent claim that is paid. It excludes hosting, staff training, integration, and delayed-customer costs.

## What failed or did not help

- **Plain accuracy:** looked excellent even for the useless all-genuine baseline, so it was rejected as the main decision metric.
- **Random splitting:** leaked repeated operational patterns across time and ignored the May policy change.
- **Text-only modelling:** missed important product, partner, inspection, and policy context.
- **Fixed keyword rules:** reduced validation performance on claims containing several issues, so they were removed.
- **Using all 40 review slots:** caught two extra frauds but created many more genuine holds; its estimated net value was far below the 20-claim queue.
- **Early post-policy validation:** performance fell sharply immediately after the May process change. This is evidence of concept drift, not something to hide; monitoring after launch is required.

## Limits and safeguards

- Historical labels are operational outcomes, not perfect ground truth.
- Partner history can reflect past process choices, so it must be monitored by partner age/type and never be the sole basis for rejection.
- The exact threshold should be recalibrated after one week of shadow use.
- A reviewer should see reasons and source fields, confirm evidence, and record an outcome.
- Production access, authentication, audit logs, and secure deployment are outside this prototype.

## Reproducibility

`python src/train.py --data-dir data` reproduces the validation, writes `artifacts/metrics.json`, saves the local model, and produces `predictions.csv`. Automated endpoint checks run with `python -m pytest -q`.
