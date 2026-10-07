# To: Ritu — Warranty claim review recommendation

## Recommendation

Run this model in **shadow mode for one week**, then use it only to rank a maximum of **20 claims per month for manual investigation**. Do not auto-reject customers and do not immediately replace the existing process.

## Why

Fraud is rare, so the board's 97% accuracy target is not useful by itself. A system that approves every claim already exceeds 97% accuracy but prevents no loss. On the latest resolved month, the proposed ranking placed 13 of 22 fraudulent claims in the first 20 reviews. Sixty-five percent of that queue was genuinely fraudulent.

Using the cost assumptions supplied by operations, those 20 reviews identified **Rs 19,099** in fraudulent claim value. After **Rs 5,200** in review-contact costs and **Rs 2,660** in goodwill costs for seven genuine customers held, the estimated net value was **Rs 11,239 for that month**. Expanding the queue to all 40 available slots caught three additional frauds and reduced estimated net value to Rs 2,359, so 20 is the better starting limit.

These numbers are validation estimates, not guaranteed savings. They exclude deployment, staff time beyond the supplied service-contact cost, and any longer-term customer impact.

## What happens next week

1. Score new claims silently while the current process continues.
2. Ask investigators to review the top 20 and record the actual outcome and reason.
3. Check precision, fraud value found, genuine-customer holds, and results by new/established partner groups.
4. Review false positives and false negatives with operations; recalibrate the threshold before any live use.
5. Keep all customer and partner data in the approved secure environment with access logs.

## Go/no-go rule

Proceed to a limited live pilot only if shadow results remain financially positive, investigators find the reasons useful, and no partner group shows an unexplained concentration of false alerts. Otherwise, keep the current workflow and retrain after collecting cleaner post-policy outcomes.
