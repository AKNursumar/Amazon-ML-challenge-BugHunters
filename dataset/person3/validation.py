"""
Validation System & Threshold Optimization Engine for Entity Resolution (Person 3).
Amazon ML Challenge 2026.

Features:
1. Group-Aware Holdout Splitting (grouped strictly by source1_id to guarantee zero entity leakage).
2. Supports customizable split ratios (e.g., 90/10 recommended by challenge, 80/20).
3. Fine-grained threshold sweeping optimizing for official Macro-averaged F0.5.
4. Separate tracking of singleton accuracy and multi-match precision/recall.
5. Automated generation of benchmark CSV tables, JSON summaries, and error logs.
"""

from collections import defaultdict
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
for p in [cur_dir, dataset_dir, repo_root]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from person3.metrics import calculate_entity_metrics, evaluate_entity_resolution
    from person3.decision import DecisionEngine
except ImportError:
    from dataset.person3.metrics import calculate_entity_metrics, evaluate_entity_resolution
    from dataset.person3.decision import DecisionEngine


class ValidationSystem:
    """
    Manages group-aware validation splitting, macro-metric evaluation,
    and decision threshold tuning.
    """

    def __init__(
        self,
        ground_truth_map: Dict[str, Set[str]],
        predictions_df: pd.DataFrame,
        random_seed: int = 42,
    ):
        """
        Args:
            ground_truth_map: Mapping from source1_id -> set of true matching IDs.
            predictions_df: DataFrame with ['source1_id', 'candidate_id', 'match_probability'].
            random_seed: Seed for reproducible group splitting.
        """
        self.ground_truth_map = ground_truth_map
        self.predictions_df = predictions_df
        self.random_seed = random_seed

        # Pre-group predictions by source1_id for fast sweeping
        self.preds_by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for s1, c_id, prob in zip(
            predictions_df["source1_id"],
            predictions_df["candidate_id"],
            predictions_df["match_probability"],
        ):
            self.preds_by_s1[s1].append((c_id, float(prob)))

        # All unique S1 entities in candidate/ground truth universe
        self.all_s1_ids = np.array(sorted(list(set(predictions_df["source1_id"].unique()))))

    def create_split(
        self, val_ratio: float = 0.10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a group-aware split on source1_id.

        Args:
            val_ratio: Proportion of S1 entities to allocate to validation (e.g. 0.10 for 90/10).

        Returns:
            (train_s1_ids, val_s1_ids)
        """
        np.random.seed(self.random_seed)
        shuffled = np.random.permutation(self.all_s1_ids)
        n_val = int(len(shuffled) * val_ratio)
        val_s1 = shuffled[:n_val]
        train_s1 = shuffled[n_val:]
        return train_s1, val_s1

    def sweep_thresholds(
        self,
        evaluated_s1_ids: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Sweep candidate probability decision thresholds to find the threshold
        maximizing the competition Macro F0.5 metric.

        Args:
            evaluated_s1_ids: Array of S1 IDs in the evaluation set.
            thresholds: List of thresholds to evaluate.

        Returns:
            (optimal_threshold, results_dataframe)
        """
        if thresholds is None:
            thresholds = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        engine = DecisionEngine()
        sweep_records = []
        best_threshold = 0.75
        best_macro_f05 = -1.0

        eval_set_ids = list(evaluated_s1_ids)

        for t in thresholds:
            # Build prediction map for current threshold
            preds_map: Dict[str, Set[str]] = {}
            for s1_id in eval_set_ids:
                cand_list = self.preds_by_s1.get(s1_id, [])
                matching = {c_id for c_id, prob in cand_list if prob >= t}
                preds_map[s1_id] = matching

            metrics = evaluate_entity_resolution(
                ground_truth_map=self.ground_truth_map,
                predictions_map=preds_map,
                evaluated_s1_ids=eval_set_ids,
                beta=0.5,
            )

            metrics["threshold"] = t
            sweep_records.append(metrics)

            if metrics["macro_f05"] > best_macro_f05:
                best_macro_f05 = metrics["macro_f05"]
                best_threshold = t

        sweep_df = pd.DataFrame(sweep_records)
        return best_threshold, sweep_df


def run_validation_experiments(
    candidate_predictions_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run comprehensive validation experiments across 90/10 and 80/20 splits,
    tuning thresholds and saving benchmark artifacts.
    """
    if output_dir is None:
        output_dir = os.path.join(cur_dir, "results")
    if candidate_predictions_path is None:
        candidate_predictions_path = os.path.join(dataset_dir, "person2", "predictions", "candidate_predictions.tsv")
    if ground_truth_path is None:
        candidate_gt = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
        if os.path.exists(candidate_gt):
            ground_truth_path = candidate_gt
        else:
            ground_truth_path = os.path.join(repo_root, "Resources", "student_resource", "dataset", "train", "train_ground_truth.tsv")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print("=" * 85)
    print("PERSON 3: VALIDATION SYSTEM & THRESHOLD OPTIMIZATION")
    print("=" * 85)

    # 1. Load predictions
    print(f"\n[1/5] Loading candidate predictions from: {candidate_predictions_path}")
    pred_df = pd.read_csv(candidate_predictions_path, sep="\t")
    print(f"  Loaded {len(pred_df):,} candidate predictions across {pred_df['source1_id'].nunique():,} S1 entities.")

    # 2. Load ground truth
    print(f"\n[2/5] Loading ground truth from: {ground_truth_path}")
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    unique_s1 = set(pred_df["source1_id"].unique())
    gt_subset = gt_df[gt_df["source1_entity_id"].isin(unique_s1)]

    gt_map: Dict[str, Set[str]] = {}
    for s1_id, m_str in zip(gt_subset["source1_entity_id"], gt_subset["matched_entity_ids"]):
        m_str = m_str.strip()
        gt_map[s1_id] = set(m_str.split(",")) if m_str else set()

    total_singletons = sum(1 for v in gt_map.values() if len(v) == 0)
    print(f"  Mapped {len(gt_map):,} entities: {len(gt_map)-total_singletons:,} non-singletons, {total_singletons:,} singletons ({total_singletons/len(gt_map)*100:.2f}%).")

    # 3. Initialize validation system
    val_sys = ValidationSystem(gt_map, pred_df, random_seed=42)

    # 4. Experiment A: 90/10 Split (Recommended by Challenge)
    print("\n[3/5] Evaluating 90/10 Train/Validation Split (10% Holdout)...")
    train_90, val_10 = val_sys.create_split(val_ratio=0.10)
    print(f"  Train: {len(train_90):,} S1 queries | Validation: {len(val_10):,} S1 queries")

    best_t_10, sweep_10_df = val_sys.sweep_thresholds(val_10)
    sweep_10_path = os.path.join(output_dir, "validation_sweep_90_10.csv")
    sweep_10_df.to_csv(sweep_10_path, index=False)
    print(f"  Saved 90/10 threshold sweep to: {sweep_10_path}")

    best_row_10 = sweep_10_df[sweep_10_df["threshold"] == best_t_10].iloc[0]
    print(f"  >>> Optimal Threshold: T = {best_t_10:.2f}")
    print(f"      Macro F0.5:        {best_row_10['macro_f05']:.4f}")
    print(f"      Macro Precision:   {best_row_10['macro_precision']:.4f}")
    print(f"      Macro Recall:      {best_row_10['macro_recall']:.4f}")
    print(f"      Singleton Acc:     {best_row_10['singleton_accuracy']:.4f}")

    # 5. Experiment B: 80/20 Split (Standard 5-Fold Equivalent Holdout)
    print("\n[4/5] Evaluating 80/20 Train/Validation Split (20% Holdout)...")
    train_80, val_20 = val_sys.create_split(val_ratio=0.20)
    print(f"  Train: {len(train_80):,} S1 queries | Validation: {len(val_20):,} S1 queries")

    best_t_20, sweep_20_df = val_sys.sweep_thresholds(val_20)
    sweep_20_path = os.path.join(output_dir, "validation_sweep_80_20.csv")
    sweep_20_df.to_csv(sweep_20_path, index=False)
    print(f"  Saved 80/20 threshold sweep to: {sweep_20_path}")

    best_row_20 = sweep_20_df[sweep_20_df["threshold"] == best_t_20].iloc[0]
    print(f"  >>> Optimal Threshold: T = {best_t_20:.2f}")
    print(f"      Macro F0.5:        {best_row_20['macro_f05']:.4f}")
    print(f"      Macro Precision:   {best_row_20['macro_precision']:.4f}")
    print(f"      Macro Recall:      {best_row_20['macro_recall']:.4f}")
    print(f"      Singleton Acc:     {best_row_20['singleton_accuracy']:.4f}")

    # Full benchmark comparison table
    comp_df = pd.DataFrame([
        {
            "Split": "90% Train / 10% Val",
            "Holdout S1 Count": len(val_10),
            "Optimal Threshold": best_t_10,
            "Macro F0.5": best_row_10["macro_f05"],
            "Macro Precision": best_row_10["macro_precision"],
            "Macro Recall": best_row_10["macro_recall"],
            "Singleton Accuracy": best_row_10["singleton_accuracy"],
            "Non-Singleton F0.5": best_row_10["non_singleton_macro_f05"],
        },
        {
            "Split": "80% Train / 20% Val",
            "Holdout S1 Count": len(val_20),
            "Optimal Threshold": best_t_20,
            "Macro F0.5": best_row_20["macro_f05"],
            "Macro Precision": best_row_20["macro_precision"],
            "Macro Recall": best_row_20["macro_recall"],
            "Singleton Accuracy": best_row_20["singleton_accuracy"],
            "Non-Singleton F0.5": best_row_20["non_singleton_macro_f05"],
        },
    ])
    comp_path = os.path.join(output_dir, "split_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[5/5] Saved split comparison to: {comp_path}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "primary_split": "90_10",
            "optimal_threshold": float(best_t_10),
            "macro_f05": float(best_row_10["macro_f05"]),
            "macro_precision": float(best_row_10["macro_precision"]),
            "macro_recall": float(best_row_10["macro_recall"]),
            "singleton_accuracy": float(best_row_10["singleton_accuracy"]),
            "split_comparison": comp_df.to_dict(orient="records"),
            "total_candidate_pairs": len(pred_df),
            "total_evaluated_s1": len(val_sys.all_s1_ids),
        }, f, indent=2)
    print(f"  Saved JSON summary to: {summary_path}")

    print("=" * 85)
    print(f"PERSON 3 VALIDATION COMPLETED IN {time.time()-t0:.2f}s")
    print("=" * 85)

    return {
        "optimal_threshold": best_t_10,
        "best_macro_f05": best_row_10["macro_f05"],
        "comparison_df": comp_df,
    }


if __name__ == "__main__":
    run_validation_experiments()
