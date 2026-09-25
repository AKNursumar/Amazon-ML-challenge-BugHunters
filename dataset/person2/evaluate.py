"""
Evaluation and Metrics Module for Entity Resolution (Person 2).
Amazon ML Challenge 2026.

Implements:
- Precision, Recall, F0.5, F1, and ROC-AUC
- Threshold tuning maximizing F0.5
- Blocking recall vs Classification recall decomposition
- False Positive and False Negative error analysis
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Calculate F-beta score.
    For beta=0.5: weights precision twice as heavily as recall.
    Formula: (1 + beta^2) * (P * R) / (beta^2 * P + R)
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute complete classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f05 = calculate_f_beta(prec, rec, beta=0.5)
    f1 = calculate_f_beta(prec, rec, beta=1.0)
    acc = (tp + tn) / max(1, len(y_true))

    auc = 0.5
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

    return {
        "precision": prec,
        "recall": rec,
        "f05": f05,
        "f1": f1,
        "accuracy": acc,
        "roc_auc": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(y_true),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep probability thresholds to find optimal threshold maximizing validation F0.5.
    Returns best_threshold, threshold_results_df.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    results = []
    best_thresh = 0.50
    best_f05 = -1.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        m = calculate_metrics(y_true, y_pred, y_prob)
        results.append({
            "threshold": thresh,
            "precision": m["precision"],
            "recall": m["recall"],
            "f05": m["f05"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        })

        if m["f05"] > best_f05:
            best_f05 = m["f05"]
            best_thresh = thresh

    res_df = pd.DataFrame(results)
    return best_thresh, res_df


def evaluate_blocking_vs_classification(
    total_ground_truth: int,
    true_matches_in_candidates: int,
    correctly_classified_matches: int,
) -> Dict[str, float]:
    """
    Decompose overall system recall into blocking recall vs classification recall.
    Formula: System Recall = Blocking Recall * Classification Recall
    """
    blocking_recall = (
        true_matches_in_candidates / total_ground_truth
        if total_ground_truth > 0
        else 0.0
    )
    classification_recall = (
        correctly_classified_matches / true_matches_in_candidates
        if true_matches_in_candidates > 0
        else 0.0
    )
    system_recall = (
        correctly_classified_matches / total_ground_truth
        if total_ground_truth > 0
        else 0.0
    )

    blocking_failures = total_ground_truth - true_matches_in_candidates
    classification_failures = (
        true_matches_in_candidates - correctly_classified_matches
    )

    return {
        "total_ground_truth": total_ground_truth,
        "true_matches_in_candidates": true_matches_in_candidates,
        "correctly_classified_matches": correctly_classified_matches,
        "blocking_recall": blocking_recall,
        "classification_recall": classification_recall,
        "system_recall": system_recall,
        "blocking_failures": blocking_failures,
        "classification_failures": classification_failures,
    }


def analyze_errors(
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    entity_lookups: Dict[str, Dict[str, object]],
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze high-confidence False Positives and False Negatives.
    Returns (fp_analysis_df, fn_analysis_df).
    """
    temp_df = val_df[["source1_id", "candidate_id", "candidate_source"]].copy()
    temp_df["label"] = y_true
    temp_df["pred"] = y_pred
    temp_df["prob"] = y_prob

    # 1. False Positives (predicted 1, actual 0), highest probability first
    fp_mask = (temp_df["label"] == 0) & (temp_df["pred"] == 1)
    fp_df = temp_df[fp_mask].sort_values(by="prob", ascending=False).head(top_n).copy()

    fp_records = []
    for _, r in fp_df.iterrows():
        s1 = entity_lookups.get(r["source1_id"], {})
        cand = entity_lookups.get(r["candidate_id"], {})

        s1_name, c_name = s1.get("raw_name", ""), cand.get("raw_name", "")
        s1_addr, c_addr = s1.get("raw_addr", ""), cand.get("raw_addr", "")

        # Categorize failure pattern
        pattern = "Unknown"
        if s1_name.lower().strip() == c_name.lower().strip():
            pattern = "Identical Name, Different Location/Branch"
        elif s1.get("country") == cand.get("country") and s1.get("bldg_no") != cand.get("bldg_no"):
            pattern = "Similar Name, Different Street/Building"
        else:
            pattern = "Common Business Name Collision"

        fp_records.append({
            "source1_id": r["source1_id"],
            "candidate_id": r["candidate_id"],
            "candidate_source": r["candidate_source"],
            "match_probability": round(r["prob"], 4),
            "source1_name": s1_name,
            "candidate_name": c_name,
            "source1_address": s1_addr,
            "candidate_address": c_addr,
            "country": s1.get("country"),
            "pattern": pattern,
        })

    # 2. False Negatives (predicted 0, actual 1), lowest probability first
    fn_mask = (temp_df["label"] == 1) & (temp_df["pred"] == 0)
    fn_df = temp_df[fn_mask].sort_values(by="prob", ascending=True).head(top_n).copy()

    fn_records = []
    for _, r in fn_df.iterrows():
        s1 = entity_lookups.get(r["source1_id"], {})
        cand = entity_lookups.get(r["candidate_id"], {})

        s1_name, c_name = s1.get("raw_name", ""), cand.get("raw_name", "")
        s1_addr, c_addr = s1.get("raw_addr", ""), cand.get("raw_addr", "")

        # Categorize failure pattern
        pattern = "Unknown"
        if not c_addr:
            pattern = "Candidate Address Missing"
        elif not any(c.isascii() for c in c_name):
            pattern = "Indic Script / Multilingual Name Mismatch"
        elif s1.get("bldg_no") and cand.get("bldg_no") and s1.get("bldg_no") != cand.get("bldg_no"):
            pattern = "Address Discrepancy / Typo in Building Number"
        else:
            pattern = "Significant Spelling / Alias Variation"

        fn_records.append({
            "source1_id": r["source1_id"],
            "candidate_id": r["candidate_id"],
            "candidate_source": r["candidate_source"],
            "match_probability": round(r["prob"], 4),
            "source1_name": s1_name,
            "candidate_name": c_name,
            "source1_address": s1_addr,
            "candidate_address": c_addr,
            "country": s1.get("country"),
            "pattern": pattern,
        })

    return pd.DataFrame(fp_records), pd.DataFrame(fn_records)
