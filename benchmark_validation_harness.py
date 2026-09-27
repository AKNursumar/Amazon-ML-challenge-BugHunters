"""
Leakage-Safe Validation Benchmark Harness for Amazon ML Challenge 2026.
Team: BugHunters

Evaluates the end-to-end Entity Resolution pipeline:
1. S1 Entity-level 80/20 train/validation split (Seed 42).
2. Includes real-world 5.6% singleton distribution.
3. Candidate generation on validation S1 WITHOUT ground-truth leakage.
4. Computes 26 Person 2 features.
5. Trains LightGBM model on Train S1 candidate pairs.
6. Evaluates official Macro F0.5, Precision, Recall, Singleton Accuracy,
   Candidate Recall Ceiling, and Compactness percentiles.
7. Computes breakdowns by Country (US, India), Source Pair (S1->S2, S1->S3),
   and Match Difficulty (Exact Name, Exact Address, Approximate).
"""

import os
import sys
import time
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(cur_dir, "dataset")
for p in [cur_dir, dataset_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from dataset.person1.normalization import (
    normalize_name,
    clean_core_name,
    normalize_address,
    normalize_country,
)
from dataset.person1.blocking import BlockingIndex, generate_blocking_keys
from dataset.person2.features import (
    ALL_FEATURES,
    load_entity_lookups,
    generate_feature_matrix,
)
from dataset.person3.metrics import evaluate_entity_resolution, calculate_entity_metrics


def build_validation_dataset(random_seed: int = 42):
    """
    Builds fixed 5,000 S1 dataset with 5.6% singletons (~280 singletons + ~4,720 matched).
    """
    print("[1/5] Ingesting source datasets and constructing benchmark universe...", flush=True)
    eval_gt_df = pd.read_csv("dataset/benchmark/eval_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    eval_s1_df = pd.read_csv("dataset/benchmark/eval_source1.tsv", sep="\t", dtype=str).fillna("")
    eval_s2_df = pd.read_csv("dataset/benchmark/eval_source2.tsv", sep="\t", dtype=str).fillna("")
    eval_s3_df = pd.read_csv("dataset/benchmark/eval_source3.tsv", sep="\t", dtype=str).fillna("")

    # Sample singletons from train dataset
    train_gt_sample = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, nrows=20000).fillna("")
    singletons_gt = train_gt_sample[train_gt_sample["matched_entity_ids"] == ""].head(280)
    singleton_s1_ids = set(singletons_gt["source1_entity_id"])

    train_s1_sample = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=20000).fillna("")
    singletons_s1 = train_s1_sample[train_s1_sample["entity_id"].isin(singleton_s1_ids)].copy()

    # Combine matched S1 (4,720) + singletons (280) = 5,000 total S1
    matched_s1 = eval_s1_df.head(4720).copy()
    matched_gt = eval_gt_df.head(4720).copy()

    s1_all = pd.concat([matched_s1, singletons_s1], ignore_index=True)
    gt_all = pd.concat([matched_gt, singletons_gt], ignore_index=True)

    print(f"  Benchmark S1 entities: {len(s1_all):,} (Singletons: {len(singletons_s1):,}, Matched: {len(matched_s1):,})", flush=True)

    # Build ground truth map
    gt_map: Dict[str, Set[str]] = {}
    gt_pairs: Set[Tuple[str, str]] = set()
    for s1_id, matches in zip(gt_all["source1_entity_id"], gt_all["matched_entity_ids"]):
        if matches.strip():
            m_set = {x.strip() for x in matches.split(",") if x.strip()}
            gt_map[s1_id] = m_set
            for m in m_set:
                gt_pairs.add((s1_id, m))
        else:
            gt_map[s1_id] = set()

    return s1_all, eval_s2_df, eval_s3_df, gt_map, gt_pairs


def run_baseline_validation(
    threshold: float = 0.80,
    random_seed: int = 42,
    val_ratio: float = 0.20,
):
    t0 = time.time()
    s1_all, s2_all, s3_all, gt_map, gt_pairs = build_validation_dataset(random_seed=random_seed)

    # 1. Normalization
    print("\n[2/5] Normalizing names, addresses, countries...", flush=True)
    for df in [s1_all, s2_all, s3_all]:
        df["norm_name"] = df["business_name"].apply(normalize_name)
        df["core_name"] = df["norm_name"].apply(clean_core_name)
        df["norm_addr"] = df["business_address"].apply(normalize_address)
        df["norm_country"] = df["country"].apply(normalize_country)

    # 2. Entity-level split on S1
    np.random.seed(random_seed)
    all_s1_ids = s1_all["entity_id"].values.copy()
    np.random.shuffle(all_s1_ids)

    n_val = int(len(all_s1_ids) * val_ratio)
    val_s1_ids = set(all_s1_ids[:n_val])
    train_s1_ids = set(all_s1_ids[n_val:])

    print(f"\n[3/5] Entity-Level Split ({int((1-val_ratio)*100)}% Train / {int(val_ratio*100)}% Val, Seed={random_seed}):", flush=True)
    print(f"  Train S1 entities: {len(train_s1_ids):,}")
    print(f"  Val S1 entities:   {len(val_s1_ids):,}")

    # 3. Build Blocking Index on S2 and S3
    print("\n[4/5] Building Blocking Inverted Index on Source 2 and Source 3...", flush=True)
    t_idx = time.time()
    index = BlockingIndex(max_block_size=1000)
    index.add_candidates(s2_all, source_name="source2")
    index.add_candidates(s3_all, source_name="source3")
    pruned = index.prune_large_blocks()
    print(f"  Inverted index built ({len(index.index):,} keys, {pruned:,} pruned) in {time.time()-t_idx:.2f}s", flush=True)

    # Generate candidates for Train and Val S1 entities
    def get_candidates_for_s1(s1_subset_df):
        s1_keys_list = generate_blocking_keys(s1_subset_df)
        records = []
        cands_per_s1 = {}
        for s1_id, keys in zip(s1_subset_df["entity_id"], s1_keys_list):
            cand_scores = defaultdict(int)
            cand_src_map = {}
            for k in keys:
                if k in index.index:
                    for cid, src in index.index[k]:
                        cand_scores[cid] += 1
                        cand_src_map[cid] = src
            sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))[:300]
            c_list = []
            for cid, _ in sorted_cands:
                records.append((s1_id, cid, cand_src_map[cid]))
                c_list.append(cid)
            cands_per_s1[s1_id] = c_list
        cand_df = pd.DataFrame(records, columns=["source1_id", "candidate_id", "candidate_source"])
        return cand_df, cands_per_s1

    s1_train_df = s1_all[s1_all["entity_id"].isin(train_s1_ids)].copy().reset_index(drop=True)
    s1_val_df = s1_all[s1_all["entity_id"].isin(val_s1_ids)].copy().reset_index(drop=True)

    train_cands_df, train_cands_map = get_candidates_for_s1(s1_train_df)
    val_cands_df, val_cands_map = get_candidates_for_s1(s1_val_df)

    print(f"  Train candidates generated: {len(train_cands_df):,} pairs across {len(train_s1_ids):,} S1s", flush=True)
    print(f"  Val candidates generated:   {len(val_cands_df):,} pairs across {len(val_s1_ids):,} S1s", flush=True)

    # Blocking Recall on Validation S1
    val_gt_pairs = {p for p in gt_pairs if p[0] in val_s1_ids}
    val_cand_pairs_set = set(zip(val_cands_df["source1_id"], val_cands_df["candidate_id"]))
    blocking_hits = len(val_gt_pairs.intersection(val_cand_pairs_set))
    candidate_recall_ceiling = blocking_hits / max(1, len(val_gt_pairs))

    # Candidate Compactness on Val
    val_cand_lens = np.array([len(val_cands_map[s1]) for s1 in val_s1_ids])
    avg_cands_per_s1 = float(np.mean(val_cand_lens))
    median_cands = float(np.median(val_cand_lens))
    p90_cands = float(np.percentile(val_cand_lens, 90))
    p95_cands = float(np.percentile(val_cand_lens, 95))
    p99_cands = float(np.percentile(val_cand_lens, 99))
    max_cands = int(np.max(val_cand_lens))

    print("\n" + "=" * 80, flush=True)
    print("VALIDATION BLOCKING PERFORMANCE:", flush=True)
    print(f"  Val Ground Truth True Pairs:  {len(val_gt_pairs):,}", flush=True)
    print(f"  True Matches in Candidates:   {blocking_hits:,} ({candidate_recall_ceiling*100:.2f}% recall ceiling)", flush=True)
    print(f"  Average Candidates / S1:      {avg_cands_per_s1:.1f}", flush=True)
    print(f"  Median Candidates / S1:       {median_cands:.1f}", flush=True)
    print(f"  90th Percentile Candidates:   {p90_cands:.1f}", flush=True)
    print(f"  95th Percentile Candidates:   {p95_cands:.1f}", flush=True)
    print(f"  99th Percentile Candidates:   {p99_cands:.1f}", flush=True)
    print(f"  Max Candidates / S1:          {max_cands}", flush=True)
    print("=" * 80, flush=True)

    # 4. Feature Extraction
    print("\n[5/5] Extracting 26 features and training LightGBM champion model...", flush=True)
    needed_s1 = set(train_cands_df["source1_id"]).union(set(val_cands_df["source1_id"]))
    needed_cands = set(train_cands_df["candidate_id"]).union(set(val_cands_df["candidate_id"]))

    lookups = load_entity_lookups(s1_all, s2_all, s3_all, needed_s1, needed_cands)

    train_feat_df = generate_feature_matrix(train_cands_df, lookups, ground_truth_pairs=gt_pairs)
    val_feat_df = generate_feature_matrix(val_cands_df, lookups, ground_truth_pairs=gt_pairs)

    X_train = train_feat_df[ALL_FEATURES].values
    y_train = train_feat_df["label"].values

    X_val = val_feat_df[ALL_FEATURES].values
    y_val = val_feat_df["label"].values

    # Train LightGBM Model
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=8,
        random_state=random_seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Predict on Validation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_feat_df["prob"] = val_probs

    # Aggregate Predictions per S1 at threshold
    val_preds_map: Dict[str, Set[str]] = {}
    val_passing = val_feat_df[val_feat_df["prob"] >= threshold]

    grouped_preds = defaultdict(list)
    for s1_id, cid, prob in zip(val_passing["source1_id"], val_passing["candidate_id"], val_passing["prob"]):
        if cid.startswith(("S2-", "S3-")):
            grouped_preds[s1_id].append((cid, prob))

    for s1_id in val_s1_ids:
        c_tuples = grouped_preds.get(s1_id, [])
        if c_tuples:
            c_tuples.sort(key=lambda x: (-x[1], x[0]))
            val_preds_map[s1_id] = {c[0] for c in c_tuples}
        else:
            val_preds_map[s1_id] = set()

    # Calculate Official Validation Metrics
    val_metrics = evaluate_entity_resolution(
        ground_truth_map=gt_map,
        predictions_map=val_preds_map,
        evaluated_s1_ids=list(val_s1_ids),
        beta=0.5,
    )

    # Detailed Breakdowns
    # Country Breakdown
    country_map = dict(zip(s1_all["entity_id"], s1_all["country"]))
    us_val_s1 = [s for s in val_s1_ids if country_map.get(s) == "US"]
    india_val_s1 = [s for s in val_s1_ids if country_map.get(s) == "India"]

    us_metrics = evaluate_entity_resolution(gt_map, val_preds_map, evaluated_s1_ids=us_val_s1, beta=0.5) if us_val_s1 else {}
    india_metrics = evaluate_entity_resolution(gt_map, val_preds_map, evaluated_s1_ids=india_val_s1, beta=0.5) if india_val_s1 else {}

    # Source Pair Breakdown (S1->S2 vs S1->S3)
    s2_gt_map = {s: {c for c in gt_map.get(s, set()) if c.startswith("S2-")} for s in val_s1_ids}
    s2_pred_map = {s: {c for c in val_preds_map.get(s, set()) if c.startswith("S2-")} for s in val_s1_ids}
    s2_metrics = evaluate_entity_resolution(s2_gt_map, s2_pred_map, evaluated_s1_ids=list(val_s1_ids), beta=0.5)

    s3_gt_map = {s: {c for c in gt_map.get(s, set()) if c.startswith("S3-")} for s in val_s1_ids}
    s3_pred_map = {s: {c for c in val_preds_map.get(s, set()) if c.startswith("S3-")} for s in val_s1_ids}
    s3_metrics = evaluate_entity_resolution(s3_gt_map, s3_pred_map, evaluated_s1_ids=list(val_s1_ids), beta=0.5)

    # Match Category Breakdown
    # Exact Name vs Exact Address vs Approximate
    name_map = dict(zip(s1_all["entity_id"], s1_all["norm_name"]))
    addr_map = dict(zip(s1_all["entity_id"], s1_all["norm_addr"]))

    s2_name_map = dict(zip(s2_all["entity_id"], s2_all["norm_name"]))
    s3_name_map = dict(zip(s3_all["entity_id"], s3_all["norm_name"]))
    cand_name_map = {**s2_name_map, **s3_name_map}

    s2_addr_map = dict(zip(s2_all["entity_id"], s2_all["norm_addr"]))
    s3_addr_map = dict(zip(s3_all["entity_id"], s3_all["norm_addr"]))
    cand_addr_map = {**s2_addr_map, **s3_addr_map}

    exact_name_s1 = []
    exact_addr_s1 = []
    approx_s1 = []

    for s1 in val_s1_ids:
        true_cands = gt_map.get(s1, set())
        if not true_cands:
            continue
        s1_n = name_map.get(s1, "")
        s1_a = addr_map.get(s1, "")

        has_exact_name = any(cand_name_map.get(c) == s1_n for c in true_cands if s1_n)
        has_exact_addr = any(cand_addr_map.get(c) == s1_a for c in true_cands if s1_a)

        if has_exact_name and has_exact_addr:
            exact_name_s1.append(s1)
        elif has_exact_name or has_exact_addr:
            exact_addr_s1.append(s1)
        else:
            approx_s1.append(s1)

    exact_metrics = evaluate_entity_resolution(gt_map, val_preds_map, evaluated_s1_ids=exact_name_s1, beta=0.5) if exact_name_s1 else {}
    semi_exact_metrics = evaluate_entity_resolution(gt_map, val_preds_map, evaluated_s1_ids=exact_addr_s1, beta=0.5) if exact_addr_s1 else {}
    approx_metrics = evaluate_entity_resolution(gt_map, val_preds_map, evaluated_s1_ids=approx_s1, beta=0.5) if approx_s1 else {}

    # Total Links and Prediction counts
    total_links = sum(len(v) for v in val_preds_map.values())
    non_empty_count = sum(1 for v in val_preds_map.values() if len(v) > 0)
    empty_count = len(val_s1_ids) - non_empty_count
    avg_links_per_s1 = total_links / len(val_s1_ids)

    print("\n" + "=" * 85, flush=True)
    print("BASELINE VALIDATION RESULTS (Threshold = 0.80):", flush=True)
    print(f"  Macro F0.5:             {val_metrics['macro_f05']:.4f}", flush=True)
    print(f"  Macro Precision:        {val_metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall:           {val_metrics['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy:     {val_metrics['singleton_accuracy']:.4f} ({val_metrics['correct_singletons']}/{val_metrics['true_singletons']})", flush=True)
    print(f"  Non-singleton F0.5:     {val_metrics['non_singleton_macro_f05']:.4f}", flush=True)
    print(f"  Total Val S1 Entities:  {len(val_s1_ids):,}", flush=True)
    print(f"  Predicted Non-Empty S1: {non_empty_count:,}", flush=True)
    print(f"  Predicted Empty S1:     {empty_count:,}", flush=True)
    print(f"  Total Predicted Links:  {total_links:,}", flush=True)
    print(f"  Average Links / S1:     {avg_links_per_s1:.3f}", flush=True)
    print("=" * 85, flush=True)

    print("\n--- BREAKDOWN BY COUNTRY ---", flush=True)
    if us_metrics:
        print(f"  US ({len(us_val_s1):,} S1s):    Macro F0.5 = {us_metrics['macro_f05']:.4f} | Prec: {us_metrics['macro_precision']:.4f} | Rec: {us_metrics['macro_recall']:.4f} | SingAcc: {us_metrics['singleton_accuracy']:.4f}", flush=True)
    if india_metrics:
        print(f"  India ({len(india_val_s1):,} S1s): Macro F0.5 = {india_metrics['macro_f05']:.4f} | Prec: {india_metrics['macro_precision']:.4f} | Rec: {india_metrics['macro_recall']:.4f} | SingAcc: {india_metrics['singleton_accuracy']:.4f}", flush=True)

    print("\n--- BREAKDOWN BY SOURCE PAIR ---", flush=True)
    print(f"  S1 -> S2: Macro F0.5 = {s2_metrics['macro_f05']:.4f} | Prec: {s2_metrics['macro_precision']:.4f} | Rec: {s2_metrics['macro_recall']:.4f}", flush=True)
    print(f"  S1 -> S3: Macro F0.5 = {s3_metrics['macro_f05']:.4f} | Prec: {s3_metrics['macro_precision']:.4f} | Rec: {s3_metrics['macro_recall']:.4f}", flush=True)

    print("\n--- BREAKDOWN BY MATCH DIFFICULTY ---", flush=True)
    if exact_metrics:
        print(f"  Exact Name & Address ({len(exact_name_s1):,} S1s): Macro F0.5 = {exact_metrics['macro_f05']:.4f} | Prec: {exact_metrics['macro_precision']:.4f} | Rec: {exact_metrics['macro_recall']:.4f}", flush=True)
    if semi_exact_metrics:
        print(f"  Exact Name or Address ({len(exact_addr_s1):,} S1s): Macro F0.5 = {semi_exact_metrics['macro_f05']:.4f} | Prec: {semi_exact_metrics['macro_precision']:.4f} | Rec: {semi_exact_metrics['macro_recall']:.4f}", flush=True)
    if approx_metrics:
        print(f"  Approximate Matches   ({len(approx_s1):,} S1s): Macro F0.5 = {approx_metrics['macro_f05']:.4f} | Prec: {approx_metrics['macro_precision']:.4f} | Rec: {approx_metrics['macro_recall']:.4f}", flush=True)

    print("=" * 85, flush=True)
    return {
        "val_metrics": val_metrics,
        "us_metrics": us_metrics,
        "india_metrics": india_metrics,
        "s2_metrics": s2_metrics,
        "s3_metrics": s3_metrics,
        "candidate_recall_ceiling": candidate_recall_ceiling,
        "compactness": {
            "avg": avg_cands_per_s1,
            "median": median_cands,
            "p90": p90_cands,
            "p95": p95_cands,
            "p99": p99_cands,
            "max": max_cands,
        },
        "total_links": total_links,
        "non_empty": non_empty_count,
        "empty": empty_count,
        "val_s1_count": len(val_s1_ids),
        "model": model,
        "val_feat_df": val_feat_df,
        "gt_map": gt_map,
        "lookups": lookups,
    }


if __name__ == "__main__":
    run_baseline_validation()
