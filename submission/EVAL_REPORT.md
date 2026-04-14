# EVAL_REPORT

## 1) Evaluation Setup

- Positive class: `spoof`
- Validation split: 25% stratified holdout from train
- Test split now includes ground truth (`label_spoof`) by design
- Models compared:
  - Rules-only
  - ML-only (Logistic Regression)
  - Hybrid (0.75 ML + 0.25 Rules)
- Threshold policy:
  - ML threshold selected from PR curve to satisfy precision target ~0.85
  - Hybrid threshold selected from PR curve to satisfy precision target ~0.88

Thresholds used:
- ML: 0.5022
- Hybrid: 0.4296
- Rules fixed: 0.30

Refit protocol used in current run:
- Final model refit dataset: `test` (with labels)
- `results.json` is produced after this test-refit stage

## 2) Core Metrics

| System | Precision | Recall | F1 |
|---|---:|---:|---:|
| Rules-only (validation) | 1.0000 | 0.2870 | 0.4460 |
| ML-only (validation) | 0.8508 | 0.9395 | 0.8929 |
| Hybrid (validation) | 0.8802 | 0.9305 | 0.9046 |
| Hybrid (refit on test) | 0.8724 | 0.9404 | 0.9051 |

Average Precision (area under PR-style ranking metric):
- ML: 0.9770
- Hybrid: 0.9768

Interpretation:
- Rules baseline is high precision but low recall (strict rules catch obvious attacks).
- ML greatly improves recall with strong precision.
- Hybrid slightly improves F1 and precision over ML-only at a small recall cost.
- Test-refit hybrid yields similar F1 with higher recall and slightly lower precision.

Important note:
- Refit-on-test is included because of the explicit workflow request in this version.
- In a production-grade evaluation protocol, this would be treated as data leakage and avoided.

## 3) PR Curve and Operating Point

- PR curve artifact: `pr_curve.png`
- Chosen operating point emphasizes reviewer trust and fraud-ops workload control.
- The hybrid threshold was chosen to keep precision near 0.88 while preserving recall above 0.93.

Metric definitions follow scikit-learn references:
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

## 4) Tiny Ablation

- Rules-only under-detects subtle spoofing because many positives do not trigger hard rules.
- ML-only captures softer multivariate patterns.
- Hybrid captures explicit rule certainty and learned soft interactions.
- Under test-refit, hybrid remains directionally consistent with validation ablation.

## 5) Weak Labeling Slice (AI-Augmentation)

Weak label from text note (`src/ai_helper.py`) compared to ground truth on a random slice:
- Precision: 1.0000
- Recall: 0.8571
- F1: 0.9231

Observation:
- Note-based weak labels are conservative but useful for bootstrapping high-confidence positives.

## 6) Error Analysis

Examples are stored in:
- `error_analysis_fp.csv`
- `error_analysis_fn.csv`

### 5 False Positives (examples)

1. Event 13131: high hybrid score despite normal label due score accumulation from network-risk context and model confidence drift.
2. Event 15005: geo/IP mismatch with high speed pattern looked suspicious though generated as non-spoof.
3. Event 1261: cold-start row (`delta_t_s=0`) with unusual context likely over-weighted by model priors.
4. Event 17904: VPN presence and background context contributed to elevated risk without attack injection.
5. Event 4315: explicit geo/IP conflict with VPN generated as legitimate edge case, flagged as spoof.

### 5 False Negatives (examples)

1. Event 11410 (`ip_geo_mismatch` attack type): weak anomaly amplitude, no hard-rule trigger.
2. Event 3066 (`sensor_inconsistency`): subtle contradiction (precision vs satellites) stayed below final threshold.
3. Event 16806 (`ip_geo_mismatch`): mismatch note exists but structured signals remained close to normal cluster.
4. Event 15938 (`sensor_inconsistency`): low satellite count conflict was mild, final hybrid score remained sub-threshold.
5. Event 3531 (`ip_geo_mismatch`): attack generated with only limited contextual inconsistency, resulting score too low.

## 7) Tuning Ideas and Trade-offs

- To reduce FPs:
  - Increase minimum confidence when `geo_ip_mismatch=1` but no VPN/proxy evidence.
  - Add per-device calibration to avoid over-flagging noisy devices.
- To reduce FNs:
  - Increase weight for sensor-conflict family or lower hybrid threshold slightly.
  - Add sequence-level model features (rolling consistency across 3-5 events).

Operational trade-off:
- Current threshold prioritizes precision to reduce manual review burden while keeping high recall.
