# AI-Driven Detection of Location Spoofing (Android)

This submission builds a reproducible synthetic-data pipeline and spoof detector with:
- Interpretable rules baseline
- Compact ML model
- Hybrid scorer (rules + ML)
- Mocked AI helper for rule proposal, weak labeling, and explanations

All required deliverables are included under this `submission` directory.

## Deliverable Structure

- `data/train.csv`
- `data/test.csv`
- `src/generate_data.py`
- `src/rules_baseline.py`
- `src/model_train_eval.py`
- `src/ai_helper.py`
- `README.md`
- `DATACARD.md`
- `AI_USAGE.md`
- `EVAL_REPORT.md`
- `design.md`
- `results.json`
- `requirements.txt`

## One-Command Run (using uv)

From repository root:

```bash
uv run --with-requirements submission/requirements.txt submission/src/generate_data.py --output-dir submission/data --train-rows 18000 --test-rows 5000 --spoof-rate 0.2 && uv run --with-requirements submission/requirements.txt submission/src/model_train_eval.py
```

This command generates/refreshes:
- `submission/data/train.csv`
- `submission/data/test.csv`
- `submission/results.json`
- `submission/pr_curve.png`
- `submission/eval_summary.json`
- `submission/error_analysis_fp.csv`
- `submission/error_analysis_fn.csv`
- `submission/gemini_rule_proposals.json`

Gemini setup (required):

```bash
export GEMINI_API_KEY="your_key_here"
```

Optional model override:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

## Notes

- Platform choice: Android
- Test set includes ground truth (`label_spoof`) in the current workflow configuration
- Model threshold is selected using a precision-target policy and PR-curve analysis
- `src/ai_helper.py` calls Gemini with prompts for `propose_rules`, `weak_label_batch`, and `explain_event`
- `weak_label_from_note` intentionally remains local deterministic logic (no Gemini call)
