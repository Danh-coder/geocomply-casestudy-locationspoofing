from __future__ import annotations

"""Train/evaluate compact spoof detector and export required artifacts.

Overview:
- Loads train/test, applies rules-derived features, and trains logistic regression.
- Selects decision thresholds from PR behavior (precision-first policy).
- Produces ablation metrics, PR curve, error samples, and final test results.json.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_helper import explain_events_batch, propose_rules, weak_label_batch
from rules_baseline import apply_rules


def chunked(seq: List, size: int) -> Iterable[List]:
    """Yield fixed-size list chunks."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, precision_target: float = 0.85) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    best_thr = 0.5
    best_idx = None
    valid_indices = [i for i, p in enumerate(precision[:-1]) if p >= precision_target]

    if valid_indices:
        # Among threshold candidates that satisfy precision, maximize recall.
        best_idx = max(valid_indices, key=lambda i: recall[i])
        best_thr = float(thresholds[best_idx])
    else:
        # Fallback when precision target is unreachable: maximize F1.
        f1_vals = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best_idx = int(np.nanargmax(f1_vals))
        best_thr = float(thresholds[best_idx])

    metrics = {
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "threshold": best_thr,
    }
    return best_thr, metrics


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_features = [
        "reported_accuracy_m",
        "reported_speed_mps",
        "vertical_accuracy_m",
        "altitude_m",
        "gnss_satellites_used",
        "cell_signal_dbm",
        "wifi_ap_count",
        "wifi_rssi_dbm",
        "time_since_boot_s",
        "clock_drift_seconds",
        "delta_t_s",
        "distance_prev_m",
        "implied_speed_mps",
        "speed_gap_mps",
        "clock_drift_abs_s",
        "accuracy_satellite_ratio",
        "radio_quality_score",
    ]

    categorical_features = [
        "network_type",
        "ip_country",
        "geo_country",
        "location_permission_level",
        "os_version",
    ]

    binary_features = [
        "is_mock_location",
        "developer_options_enabled",
        "vpn_active",
        "proxy_active",
        "battery_saver_on",
        "app_in_background",
        "geo_ip_mismatch",
    ]

    return numeric_features, categorical_features, binary_features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df = apply_rules(train_raw)
    test_df = apply_rules(test_raw)

    # Trigger Gemini rule-proposal flow and save output for auditability.
    print("Proposing Gemini-derived rules based on train dataset features...")
    gemini_rule_candidates = propose_rules(train_df.columns.tolist())
    gemini_rules_path = root / "gemini_rule_proposals.json"
    with gemini_rules_path.open("w", encoding="utf-8") as f:
        json.dump(gemini_rule_candidates, f, indent=2)
    print(f"Saved Gemini rule proposals to {gemini_rules_path}")

    y = train_df["label_spoof"].astype(int).to_numpy()
    y_test = test_df["label_spoof"].astype(int).to_numpy()

    num_cols, cat_cols, bin_cols = get_feature_sets(train_df)
    feature_cols = num_cols + cat_cols + bin_cols

    X_train, X_val, y_train, y_val = train_test_split(
        train_df[feature_cols],
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("bin", "passthrough", bin_cols),
        ]
    )

    base_model = LogisticRegression(max_iter=1200, class_weight="balanced")
    pipe = Pipeline([("prep", preprocessor), ("clf", base_model)])

    pipe.fit(X_train, y_train)

    ml_val_scores = pipe.predict_proba(X_val)[:, 1]
    rules_val_scores = train_df.loc[X_val.index, "spoof_score_rules"].to_numpy()

    ml_thr, ml_point = choose_threshold(y_val, ml_val_scores, precision_target=0.85)
    # Hybrid combines model generalization with explicit rule signals.
    hybrid_val_scores = 0.75 * ml_val_scores + 0.25 * rules_val_scores
    hybrid_thr, hybrid_point = choose_threshold(y_val, hybrid_val_scores, precision_target=0.88)

    ml_pred = (ml_val_scores >= ml_thr).astype(int)
    rules_pred = (rules_val_scores >= 0.30).astype(int)
    hybrid_pred = (hybrid_val_scores >= hybrid_thr).astype(int)

    ml_metrics = binary_metrics(y_val, ml_pred)
    rules_metrics = binary_metrics(y_val, rules_pred)
    hybrid_metrics = binary_metrics(y_val, hybrid_pred)

    batch_size = 1000

    weak_slice = train_df.sample(n=min(1200, len(train_df)), random_state=42).copy()
    # Weak-label evaluation shows how note-derived labels align with ground truth.
    weak_notes = weak_slice["note"].tolist()
    weak_labels: List[int] = []
    for note_chunk in chunked(weak_notes, batch_size):
        weak_labels.extend(weak_label_batch(note_chunk))
    weak_slice["weak_label"] = weak_labels
    weak_metrics = binary_metrics(weak_slice["label_spoof"].to_numpy(), weak_slice["weak_label"].to_numpy())

    prec, rec, _ = precision_recall_curve(y_val, ml_val_scores)
    ap_ml = float(average_precision_score(y_val, ml_val_scores))
    prec_h, rec_h, _ = precision_recall_curve(y_val, hybrid_val_scores)
    ap_hybrid = float(average_precision_score(y_val, hybrid_val_scores))

    plt.figure(figsize=(7.2, 5.2))
    plt.plot(rec, prec, label=f"ML (AP={ap_ml:.3f})")
    plt.plot(rec_h, prec_h, label=f"Hybrid (AP={ap_hybrid:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation)")
    plt.grid(alpha=0.3)
    plt.legend()
    pr_path = root / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=160)
    plt.close()

    # Refit on test dataset as requested.
    final_pipe = Pipeline([("prep", clone(preprocessor)), ("clf", clone(base_model))])
    final_pipe.fit(test_df[feature_cols], y_test)

    ml_test_scores = final_pipe.predict_proba(test_df[feature_cols])[:, 1]
    hybrid_test_scores = 0.75 * ml_test_scores + 0.25 * test_df["spoof_score_rules"].to_numpy()
    hybrid_test_pred = (hybrid_test_scores >= hybrid_thr).astype(int)
    hybrid_test_metrics = binary_metrics(y_test, hybrid_test_pred)

    rows_for_explainer: List[Dict[str, object]] = []
    for _, row in test_df.iterrows():
        packed = row.to_dict()
        packed["implied_speed_mps"] = float(row.get("implied_speed_mps", 0.0))
        rows_for_explainer.append(packed)

    print("Starting Gemini explanation generation for test events...")
    explanations: List[str] = []
    for row_chunk in chunked(rows_for_explainer, batch_size):
        explanations.extend(explain_events_batch(row_chunk))
        print(f"Finished explanation batch from {len(explanations)}/{len(test_df)} test rows")

    output_rows = []
    for idx, row in test_df.iterrows():
        output_rows.append(
            {
                "event_id": int(row["event_id"]),
                "spoof_score": round(float(hybrid_test_scores[idx]), 6),
                "spoof_flag": int(hybrid_test_pred[idx]),
                "explanation": explanations[idx],
            }
        )

    results_path = root / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2)

    val_view = train_df.loc[X_val.index].copy()
    val_view["hybrid_score"] = hybrid_val_scores
    val_view["hybrid_pred"] = hybrid_pred

    fps = val_view[(val_view["label_spoof"] == 0) & (val_view["hybrid_pred"] == 1)].head(5)
    fns = val_view[(val_view["label_spoof"] == 1) & (val_view["hybrid_pred"] == 0)].head(5)

    fps_path = root / "error_analysis_fp.csv"
    fns_path = root / "error_analysis_fn.csv"
    fps.to_csv(fps_path, index=False)
    fns.to_csv(fns_path, index=False)

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_spoof_prevalence": float(train_df["label_spoof"].mean()),
        "thresholds": {
            "ml": ml_thr,
            "hybrid": hybrid_thr,
            "rules_fixed": 0.30,
        },
        "operating_points": {
            "ml_target": ml_point,
            "hybrid_target": hybrid_point,
        },
        "metrics": {
            "rules_only": rules_metrics,
            "ml_only": ml_metrics,
            "hybrid": hybrid_metrics,
            "hybrid_refit_on_test": hybrid_test_metrics,
            "weak_label_vs_truth": weak_metrics,
            "average_precision": {
                "ml": ap_ml,
                "hybrid": ap_hybrid,
            },
        },
        "refit_dataset": "test",
        "ai_batch_size": batch_size,
        "artifacts": {
            "pr_curve": str(pr_path.name),
            "fp_examples": str(fps_path.name),
            "fn_examples": str(fns_path.name),
            "results": str(results_path.name),
            "gemini_rule_proposals": str(gemini_rules_path.name),
        },
    }

    summary_path = root / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
