# Error costs: false alarm vs. missed negative (draft, 2026-11-02)

Basis: confusion matrix of the second-iteration fine-tuned model on its own shared test (1,787 DE/EN sentences, `reports/finetuned_confusion_matrix.csv`). Stakeholders: Newsrooms, PR & Communications, Analysts. Cost weights below are assumptions to discuss, not measured.

| | Count | Rate |
| --- | ---: | ---: |
| Negative texts in test | 464 | |
| Caught as negative (recall) | 344 | 74.1 % |
| Missed negatives (predicted neutral/positive) | 120 | 25.9 % |
| Predicted negative | 494 | |
| False alarms (not negative, flagged negative) | 150 | 30.4 % of alerts, 11.3 % of non-negative texts |

- **Missed negative = expensive.** PR & Communications reacts late to a critical report; Analysts miss a risk signal. Damage grows with time and reach.
- **False alarm = cheap but not free.** A person reads one extra article (about 1–3 min). Too many alarms cause alert fatigue and the tool gets ignored.
- **Assumption for discussion:** one missed negative costs about 5–10 false alarms. With that ratio the current model (120 misses vs. 150 false alarms) is tuned too conservatively for alerting use.
- **Lever without retraining:** lower the decision threshold for `negative` (score margin) and report recall/precision for negative on the shared test; pick the point that keeps false alarms per day acceptable for a newsroom.
- **Human in the loop stays:** the model ranks and pre-filters; a person decides before any reaction. This caps the cost of a false alarm.

Threshold curve: `python -m second_path.threshold_curve` writes recall/precision/false alarms for `negative` per margin (0 to 0.40) to `reports/negative_threshold_curve_val.csv`. The margin is picked on the validation split only; the test split stays locked by the evaluation protocol and is reported once for the chosen margin. Needs the fine-tuned model locally (not on the repo/box), so it is not run yet.

Open: confirm cost ratio with a stakeholder example (e.g. daily article volume of one newsroom); run the curve once on the validation split.
