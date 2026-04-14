# AI_USAGE

## 1) AI Tools Used

- GitHub Copilot (GPT-5.3-Codex) for code drafting/refinement
- Google Gemini API (prompt-based calls via `src/ai_helper.py`)
- Targeted web research to ground feature feasibility and metrics definitions

## 2) Prompt Log

### Prompt 1

Purpose: SDK feature brainstorming

Prompt:

```text
Propose SDK-collectable Android telemetry fields for GPS spoofing detection; include realistic value ranges and collection feasibility.
```

Accepted output:
- is_mock_location, accuracy/speed/bearing-related fields
- GNSS/cell/Wi-Fi quality context features
- IP-vs-geo mismatch and clock drift concepts

Edited output:
- Narrowed to policy-feasible and assignment-scoped fields
- Added explicit field-level rationale in DATACARD.md

### Prompt 2

Purpose: Synthetic data generation strategy

Prompt:

```text
Draft Python to synthesize per-device tracks with realistic jitter and spoof injections for teleportation, IP/Geo mismatch, and time drift.
```

Accepted output:
- Per-device temporal tracks and movement mode simulation
- Multi-attack spoof taxonomy with configurable prevalence

Edited output:
- Added deterministic seed controls for reproducibility
- Tuned positive prevalence to the required 10-30% range

### Prompt 3

Purpose: Interpretable rule baseline design

Prompt:

```text
Given this schema and positive/negative examples, propose interpretable spoofing rules with threshold candidates and likely false positives.
```

Accepted output:
- Impossible speed rule
- Mock-location rule
- Geo/IP conflict rule with VPN or proxy context

Edited output:
- Reweighted rules and tuned threshold for non-zero recall in ablation

### Prompt 4

Purpose: Thresholding and PR operating point

Prompt:

```text
Choose a production threshold for spoof scoring using precision/recall trade-offs and PR analysis.
```

Accepted output:
- Precision-first threshold policy
- PR-based threshold selection for ML and hybrid variants

Edited output:
- Used separate precision targets for ML and hybrid to match review workload goals

### Prompt 5

Purpose: Event explanation template

Prompt:

```text
Explain in 2-3 sentences why this event is likely spoofed using the strongest anomalous fields.
```

Accepted output:
- Concise reason templates based on top anomaly evidence

Edited output:
- Implemented Gemini-backed explainer call in `src/ai_helper.py` with local fallback behavior

### Prompt 6

Purpose: Evaluation protocol adjustment

Prompt:

```text
Update the pipeline so test.csv includes ground-truth labels and refit the final model on the test dataset before producing final scores.
```

Accepted output:
- test.csv retains label_spoof
- final refit stage is trained on test data

Edited output:
- Added explicit reporting fields in eval summary: hybrid_refit_on_test and refit_dataset
- Regenerated datasets and reran evaluation artifacts after the protocol change

### Prompt 7

Purpose: Gemini integration in evaluation runtime

Prompt:

```text
Replace mocked AI helper behavior with true Gemini prompt calls and ensure model_train_eval.py actively uses those calls during evaluation.
```

Accepted output:
- Implemented real Gemini HTTP calls in `propose_rules`, `weak_label_batch`, and `explain_event`
- Added Gemini rule proposal call and artifact generation in `model_train_eval.py` (`gemini_rule_proposals.json`)

Edited output:
- Kept `weak_label_from_note` as local deterministic logic by design
- Added environment-based configuration (`GEMINI_API_KEY`, optional `GEMINI_MODEL`)

### Prompt 8

Purpose: Scale AI helper runtime calls

Prompt:

```text
Call AI helper functions in batches of 1,000 rows instead of row-by-row calls.
```

Accepted output:
- `model_train_eval.py` now batches weak-label requests in chunks of 1,000 rows
- `model_train_eval.py` now batches explanation generation in chunks of 1,000 rows via `explain_events_batch`

Edited output:
- Added `ai_batch_size` to evaluation summary for observability
- Kept compatibility wrapper `explain_event` while runtime uses batched path

## 3) Weak Labeling (AI-Augmentation)

- Implemented deterministic weak labeler from note text (weak_label_from_note) as local logic without Gemini call.
- Gemini prompt calls are used in `propose_rules`, `weak_label_batch`, and `explain_event`.
- Runtime batching policy: AI helper calls are executed in chunks of 1,000 rows.
- Compared weak labels with synthetic ground truth in evaluation artifacts.

## 4) Reproducibility and Safety

- Gemini-backed flows require `GEMINI_API_KEY` in environment.
- Local fallback is preserved for selected cases to keep the pipeline robust when the API is unavailable.
