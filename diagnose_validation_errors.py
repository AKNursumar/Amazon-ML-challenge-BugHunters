"""
Comprehensive Error Analysis and Failure Diagnosis.
Classifies and quantifies error modes across the validation set.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(cur_dir, "dataset")
for p in [cur_dir, dataset_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from benchmark_validation_harness import run_baseline_validation

def diagnose_errors():
    print("=" * 85)
    print("RUNNING IN-DEPTH VALIDATION ERROR ANALYSIS")
    print("=" * 85)

    base = run_baseline_validation(threshold=0.80, random_seed=42, val_ratio=0.20)
    val_feat_df = base["val_feat_df"]
    gt_map = base["gt_map"]
    lookups = base["lookups"]
    val_s1_ids = set(val_feat_df["source1_id"].unique())

    # Categorize entities and pairs
    # 1. Blocking Misses vs Model False Negatives
    blocking_misses = []
    model_false_negatives = []
    threshold_false_negatives = [] # 0.50 <= prob < 0.80
    hard_model_false_negatives = [] # prob < 0.50

    val_cand_pairs_dict = defaultdict(dict)
    for s1_id, cid, prob in zip(val_feat_df["source1_id"], val_feat_df["candidate_id"], val_feat_df["prob"]):
        val_cand_pairs_dict[s1_id][cid] = prob

    for s1_id in val_s1_ids:
        true_cands = gt_map.get(s1_id, set())
        for true_cid in true_cands:
            if true_cid not in val_cand_pairs_dict[s1_id]:
                blocking_misses.append((s1_id, true_cid))
            else:
                prob = val_cand_pairs_dict[s1_id][true_cid]
                if prob < 0.80:
                    model_false_negatives.append((s1_id, true_cid, prob))
                    if prob >= 0.50:
                        threshold_false_negatives.append((s1_id, true_cid, prob))
                    else:
                        hard_model_false_negatives.append((s1_id, true_cid, prob))

    # 2. False Positives Breakdown
    singleton_false_positives = []
    non_singleton_false_positives = []
    branch_collisions = []
    common_name_collisions = []
    address_mismatches = []

    passing_df = val_feat_df[val_feat_df["prob"] >= 0.80]
    for s1_id, cid, prob in zip(passing_df["source1_id"], passing_df["candidate_id"], passing_df["prob"]):
        true_cands = gt_map.get(s1_id, set())
        if cid not in true_cands:
            s1_rec = lookups.get(s1_id, {})
            c_rec = lookups.get(cid, {})

            s1_n, c_n = s1_rec.get("core_name", ""), c_rec.get("core_name", "")
            s1_a, c_a = s1_rec.get("norm_addr", ""), c_rec.get("norm_addr", "")

            if len(true_cands) == 0:
                singleton_false_positives.append((s1_id, cid, prob, s1_n, c_n, s1_a, c_a))
            else:
                non_singleton_false_positives.append((s1_id, cid, prob, s1_n, c_n, s1_a, c_a))

            # Branch check (e.g. north vs south, unit differences)
            s1_toks = set(s1_n.split())
            c_toks = set(c_n.split())
            diff_toks = (s1_toks ^ c_toks) - {"and", "the", "for"}
            if any(t in {"north", "south", "east", "west", "central", "main", "branch", "phase", "sector"} for t in diff_toks):
                branch_collisions.append((s1_id, cid, prob, s1_n, c_n))

    print("\n" + "=" * 80)
    print("ERROR QUANTIFICATION SUMMARY:")
    print("=" * 80)
    print(f"Total True Pairs in Validation:              {sum(len(gt_map.get(s, set())) for s in val_s1_ids):,}")
    print(f"1. Blocking Misses (Recall Ceiling Loss):     {len(blocking_misses):,} ({len(blocking_misses)/max(1, sum(len(gt_map.get(s, set())) for s in val_s1_ids))*100:.2f}%)")
    print(f"2. Model False Negatives (In Cands, prob<0.8):{len(model_false_negatives):,}")
    print(f"   - Threshold Borderline (0.50 <= prob < 0.80): {len(threshold_false_negatives):,}")
    print(f"   - Hard Misses (prob < 0.50):                 {len(hard_model_false_negatives):,}")
    print(f"3. False Positives (Total):                   {len(singleton_false_positives) + len(non_singleton_false_positives):,}")
    print(f"   - False Merges on True Singletons:         {len(singleton_false_positives):,} ({len(set(x[0] for x in singleton_false_positives))} singleton entities corrupted!)")
    print(f"   - False Positives on Matched Entities:     {len(non_singleton_false_positives):,}")
    print(f"4. Branch / Directional Collisions:           {len(branch_collisions):,}")
    print("=" * 80)

    # Print representative samples of Blocking Misses
    print("\n--- SAMPLE BLOCKING MISSES (True matches not retrieved by blocking) ---")
    for s1_id, cid in blocking_misses[:5]:
        s1_rec = lookups.get(s1_id, {})
        c_rec = lookups.get(cid, {})
        print(f"S1 ({s1_id}): '{s1_rec.get('raw_name')}' | Addr: '{s1_rec.get('raw_addr')}'")
        print(f"Cand ({cid}): '{c_rec.get('raw_name')}' | Addr: '{c_rec.get('raw_addr')}'")
        print("-" * 50)

    # Print representative samples of False Merges on Singletons
    print("\n--- SAMPLE FALSE MERGES ON SINGLETONS (Singletons wrongly predicted as match) ---")
    for s1_id, cid, prob, s1_n, c_n, s1_a, c_a in singleton_false_positives[:5]:
        print(f"Singleton S1 ({s1_id}): '{s1_n}' | Addr: '{s1_a}'")
        print(f"Wrong Cand ({cid}, prob={prob:.4f}): '{c_n}' | Addr: '{c_a}'")
        print("-" * 50)

    # Print representative samples of Hard Model False Negatives
    print("\n--- SAMPLE HARD MODEL FALSE NEGATIVES (Candidates with prob < 0.50) ---")
    for s1_id, cid, prob in hard_model_false_negatives[:5]:
        s1_rec = lookups.get(s1_id, {})
        c_rec = lookups.get(cid, {})
        print(f"S1 ({s1_id}): '{s1_rec.get('raw_name')}' | Addr: '{s1_rec.get('raw_addr')}'")
        print(f"Cand ({cid}, prob={prob:.4f}): '{c_rec.get('raw_name')}' | Addr: '{c_rec.get('raw_addr')}'")
        print("-" * 50)

if __name__ == "__main__":
    diagnose_errors()
