# Task 2 V3 submission draft

## What did you build, and what business outcome does it move? State the number and the money.

I built a warranty-claim decision-support service that assigns a fraud-risk score to each claim, explains the main signals, and sends only the highest-risk cases to a human reviewer. On a 713-claim June temporal holdout containing 22 frauds, the top-20 queue found 13 frauds (65% queue precision and 59.1% recall). Those reviews identified Rs 19,099 of fraudulent claim value and produced an estimated Rs 11,239 net monthly value after the supplied Rs 260 review cost and Rs 380 goodwill cost for genuine claims held. The tool supports investigation; it never auto-rejects a customer.

## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

I expect approximately 0.60 PR-AUC. I estimated it using a strict time split: training on resolved claims before June 2026 and validating on all 713 resolved June claims. That holdout achieved 0.650 PR-AUC, and I reduced the expectation because July-September is later and the 1 May auto-approval change created visible concept drift. PR-AUC is the main metric because only about 1.27% of the deduplicated labelled claims are fraudulent; ordinary accuracy can exceed 97% by predicting every claim as genuine and stopping no fraud.

## How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

The untouched June holdout had 713 claims and 22 frauds. The model achieved 0.650 PR-AUC and 0.933 ROC-AUC. In the operational top-20 queue it found 13 frauds; 7 of the 20 alerts were genuine and it missed 9 of the 22 frauds. The equivalent whole-holdout error rate at that fixed queue was 2.24%, but this accuracy figure is secondary because the classes are extremely imbalanced. It is weakest around new or low-volume partners with little comparable post-policy history, borderline claims lacking inspection evidence, and patterns that changed immediately after the May policy change. I also checked exact output shape, unique IDs, score range, sample order, clean-start fallback, and endpoint error handling.

## Did you change, narrow, or push back on the client's ask? What, when, and why.

Yes. I did not optimise for or promise the board's 97% accuracy because an all-genuine baseline already exceeds it. After the initial data audit I changed the target to ranking a small review queue and measured PR-AUC, precision/recall, and rupee impact. I also narrowed the action from automatic fraud blocking to a 20-claim human-review queue. Although operations can handle 40 reviews, validation showed that 40 caught only three additional frauds while reducing estimated net value from Rs 11,239 to Rs 2,359.

## What is wrong with what you are handing us, or with the data we handed you? Be specific.

The historical outcome is not clean ground truth: unresolved CRM claims are blank, while old Zoho could not store blanks and may have written unresolved cases as 0. The training data contains resubmitted duplicate claim IDs, which I deduplicated before splitting. The 1 May sub-Rs 2,000 auto-approval policy caused concept drift and reduced inspection evidence, so older patterns are not fully representative. Serial formatting is inconsistent and had to be normalised. Partner history may reproduce past investigation choices and needs fairness monitoring, especially for new partners. Raw client data and the trained artifact cannot be placed in the public repository under the supplied policy; only the explicitly required two-column prediction deliverable is included. Finally, the prototype lacks production authentication, audit storage, monitoring, and deployment hardening.

## What did you deliberately leave out, and why that rather than something else?

I left out automatic rejection because the labels and policy regime are not reliable enough for an irreversible customer decision. I did not add a deep neural network or paid LLM call because the data is small/tabular, explanations matter, and a simpler model starts reliably at zero API cost. I removed fixed keyword overrides after they harmed temporal validation on multi-issue claims. I also left authentication, CRM write-back, and cloud deployment out of this short prototype so I could prioritise reproducible scoring, honest validation, a working endpoint, and privacy-safe delivery.

## Anything you built or found that nobody asked for?

I added a cost-aware comparison of 20 versus 40 reviews, a decision explanation for each scored claim, a no-paid-key demo fallback so the screen starts on a clean machine, output-contract checks, and privacy guardrails that prevent the raw supplied data and trained artifact from entering Git. I also generated a SHA-256 manifest so the delivered prediction file can be verified.

## What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

I used OpenAI Codex as a development assistant for initial code scaffolding, debugging, test suggestions, and reviewing documentation. It helped speed up repetitive implementation and surface checks I then verified against the files and executable results. I did not use an AI model inside the scoring product and used no paid model API. Suggestions that relied on ordinary accuracy, random splitting, or fixed keyword rules were discarded because they did not survive the temporal and business-cost checks. Screen recording: `[ADD PUBLIC/VIEW-ONLY GOOGLE DRIVE LINK]`.

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. Put the authorised client files in `data/` and run `python src/train.py --data-dir data`; this recreates the private model, metrics, and required `predictions.csv`. Do not commit the raw files or trained artifact.
2. Run `python src/service.py` and open `http://127.0.0.1:8520`; `POST /api/predict` scores one JSON claim. The public copy starts in clearly labelled demo-fallback mode if the private artifact is absent.
3. The current action is manual review of the highest-ranked 20 claims, never automatic rejection. Recheck threshold economics and partner-group false alerts after one week of shadow use.

## Honest hours spent

`[ADD ONE NUMBER AFTER THE RECORDING AND FINAL REVIEW]`

## GitHub Repo Link

`[ADD PUBLIC REPOSITORY URL]`

## What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)? Show the arithmetic. If you used no paid calls, say so.

The prototype makes no paid API or model calls, so its marginal paid-call cost is Rs 0 per prediction. At 750 claims: 750 x Rs 0 = Rs 0 per month in paid inference calls. Normal hosting and staff-review costs are separate. If 20 claims are reviewed, the supplied service-contact cost is 20 x Rs 260 = Rs 5,200 per month, plus Rs 380 for each genuine claim temporarily held. On the June validation queue that goodwill component was 7 x Rs 380 = Rs 2,660.
