"""
Official Evaluation Metrics for Business Entity Resolution (Person 3).
Amazon ML Challenge 2026.

Implements the official macro-averaged F0.5 evaluation metric:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed per Source 1 entity, then averaged across all Source 1 entities in the evaluation set.
Includes exact handling for singletons (entities with zero matches in ground truth):
- Correctly predicting empty for a singleton scores 1.0
- False merge on a singleton scores 0.0
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Calculate F-beta score for a single entity or set of predictions.
    Weights precision 2x over recall when beta=0.5.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def calculate_entity_metrics(
    true_ids: Set[str], pred_ids: Set[str], beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate (Precision, Recall, F_beta) for a single Source 1 entity.

    Handles singletons and multiple matches according to competition rules:
    - If true_ids is empty (Singleton):
        - If pred_ids is empty: Precision=1.0, Recall=1.0, F0.5=1.0 (Full credit)
        - If pred_ids is non-empty: Precision=0.0, Recall=1.0, F0.5=0.0 (False merge penalty)
    - If true_ids is non-empty:
        - If pred_ids is empty: Precision=0.0, Recall=0.0, F0.5=0.0
        - If pred_ids is non-empty:
            tp = len(pred_ids & true_ids)
            prec = tp / len(pred_ids)
            rec = tp / len(true_ids)
            f_beta = calculate_f_beta(prec, rec, beta=beta)
    """
    is_singleton = len(true_ids) == 0

    if is_singleton:
        if len(pred_ids) == 0:
            # Correct singleton prediction
            return 1.0, 1.0, 1.0
        else:
            # False merge on singleton
            return 0.0, 1.0, 0.0

    # Non-singleton true entity
    if len(pred_ids) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred_ids.intersection(true_ids))
    prec = tp / len(pred_ids)
    rec = tp / len(true_ids)
    f_beta = calculate_f_beta(prec, rec, beta=beta)
    return prec, rec, f_beta


def evaluate_entity_resolution(
    ground_truth_map: Dict[str, Set[str]],
    predictions_map: Dict[str, Set[str]],
    evaluated_s1_ids: Optional[List[str]] = None,
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    Compute official macro-averaged entity resolution metrics across all evaluated S1 entities.

    Args:
        ground_truth_map: Mapping from source1_entity_id -> set of true matching candidate IDs.
        predictions_map: Mapping from source1_entity_id -> set of predicted matching candidate IDs.
        evaluated_s1_ids: Optional list of S1 entity IDs to restrict evaluation to.
        beta: Weight on precision vs recall (default 0.5 for F0.5).

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    if evaluated_s1_ids is None:
        evaluated_s1_ids = sorted(list(ground_truth_map.keys()))

    f_beta_list = []
    precision_list = []
    recall_list = []

    singleton_f05_list = []
    non_singleton_f05_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    true_singletons = 0
    correct_singletons = 0
    false_merges_on_singletons = 0

    for s1_id in evaluated_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())

        prec, rec, f05 = calculate_entity_metrics(true_set, pred_set, beta=beta)

        precision_list.append(prec)
        recall_list.append(rec)
        f_beta_list.append(f05)

        if len(true_set) == 0:
            true_singletons += 1
            singleton_f05_list.append(f05)
            if len(pred_set) == 0:
                correct_singletons += 1
            else:
                false_merges_on_singletons += 1
                total_fp += len(pred_set)
        else:
            non_singleton_f05_list.append(f05)
            tp = len(pred_set.intersection(true_set))
            total_tp += tp
            total_fp += len(pred_set - true_set)
            total_fn += len(true_set - pred_set)

    macro_f05 = float(np.mean(f_beta_list)) if f_beta_list else 0.0
    macro_prec = float(np.mean(precision_list)) if precision_list else 0.0
    macro_rec = float(np.mean(recall_list)) if recall_list else 0.0

    singleton_acc = (
        float(correct_singletons / true_singletons) if true_singletons > 0 else 1.0
    )
    non_singleton_macro_f05 = (
        float(np.mean(non_singleton_f05_list)) if non_singleton_f05_list else 0.0
    )

    # Micro pairwise precision / recall
    micro_prec = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_rec = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f05 = calculate_f_beta(micro_prec, micro_rec, beta=beta)

    return {
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "singleton_accuracy": round(singleton_acc, 4),
        "non_singleton_macro_f05": round(non_singleton_macro_f05, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f05": round(micro_f05, 4),
        "total_evaluated_s1": len(evaluated_s1_ids),
        "true_singletons": true_singletons,
        "correct_singletons": correct_singletons,
        "false_merges_on_singletons": false_merges_on_singletons,
        "total_tp_pairs": total_tp,
        "total_fp_pairs": total_fp,
        "total_fn_pairs": total_fn,
    }
