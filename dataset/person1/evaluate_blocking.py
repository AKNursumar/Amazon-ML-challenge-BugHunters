"""
Evaluation Module for Blocking Strategies.
Amazon ML Challenge 2026.

Measures:
- Candidate Recall: (True matches found in candidate set) / (Total true matches)
- Candidate Count: Total number of candidate pairs generated
- Average Candidates per S1: Candidate Count / S1 records count
- Missed Matches Analysis: Diagnostic logging of missed true pairs
"""

import sys
import io
import time
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from .candidate_generation import generate_candidates
from .data_loader import get_true_pairs, load_ground_truth, load_source


def evaluate_candidate_set(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    s1_count: int,
    strategy_name: str = "Strategy",
) -> Dict[str, float]:
    """
    Compute blocking quality metrics against ground truth.

    Returns dict with:
    - strategy: name of strategy
    - recall: true matches in candidates / total true matches
    - candidate_count: total candidates
    - avg_candidates_per_s1: candidate_count / s1_count
    - true_matches_found: count of true matches in candidate set
    - total_true_matches: count of total true matches in ground truth
    """
    true_pairs = get_true_pairs(ground_truth_df)
    total_true = len(true_pairs)

    if total_true == 0:
        return {
            "strategy": strategy_name,
            "recall": 0.0,
            "candidate_count": len(candidate_pairs_df),
            "avg_candidates_per_s1": len(candidate_pairs_df) / max(1, s1_count),
            "true_matches_found": 0,
            "total_true_matches": 0,
        }

    # Create candidate pairs set for O(1) membership check
    cand_pairs = set(zip(candidate_pairs_df["source1_id"], candidate_pairs_df["candidate_id"]))
    hits = len(cand_pairs.intersection(true_pairs))
    recall = hits / total_true
    cand_count = len(candidate_pairs_df)
    avg_per_s1 = cand_count / max(1, s1_count)

    return {
        "strategy": strategy_name,
        "recall": recall,
        "candidate_count": cand_count,
        "avg_candidates_per_s1": avg_per_s1,
        "true_matches_found": hits,
        "total_true_matches": total_true,
    }


def analyze_missed_matches(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    max_examples: int = 10,
):
    """
    Identify and diagnose true matches that failed to be captured in candidate generation.
    Prints original and normalized fields to pinpoint failure reasons.
    """
    true_pairs = get_true_pairs(ground_truth_df)
    cand_pairs = set(zip(candidate_pairs_df["source1_id"], candidate_pairs_df["candidate_id"]))
    missed = list(true_pairs - cand_pairs)

    print(f"\n{'='*80}")
    print(f"MISSED MATCHES DIAGNOSTICS: {len(missed)} / {len(true_pairs)} missed ({len(missed)/len(true_pairs)*100:.2f}%)")
    print(f"{'='*80}")

    if not missed:
        print("Outstanding! Zero missed matches!")
        return

    sample_missed = missed[:max_examples]
    needed_s1 = {m[0] for m in sample_missed}
    needed_cand = {m[1] for m in sample_missed}

    s1_map = source1_df[source1_df["entity_id"].isin(needed_s1)].set_index("entity_id").to_dict("index")
    s2_map = source2_df[source2_df["entity_id"].isin(needed_cand)].set_index("entity_id").to_dict("index") if source2_df is not None else {}
    s3_map = source3_df[source3_df["entity_id"].isin(needed_cand)].set_index("entity_id").to_dict("index") if source3_df is not None else {}

    for idx, (s1_id, cand_id) in enumerate(sample_missed):
        s1 = s1_map.get(s1_id, {})
        cand = s2_map.get(cand_id) or s3_map.get(cand_id) or {}

        print(f"\n--- Missed Example #{idx+1} ---")
        print(f"S1 ID [{s1_id}] ({s1.get('country')})")
        print(f"  Orig Name: {s1.get('business_name')}")
        print(f"  Norm Name: {s1.get('norm_name')} | Core: {s1.get('core_name')}")
        print(f"  Orig Addr: {s1.get('business_address')}")
        print(f"  Norm Addr: {s1.get('norm_addr')}")

        print(f"True Match ID [{cand_id}] ({cand.get('country')})")
        print(f"  Orig Name: {cand.get('business_name')}")
        print(f"  Norm Name: {cand.get('norm_name')} | Core: {cand.get('core_name')}")
        print(f"  Orig Addr: {cand.get('business_address')}")
        print(f"  Norm Addr: {cand.get('norm_addr')}")
