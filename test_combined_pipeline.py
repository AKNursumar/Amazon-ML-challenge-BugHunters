"""
Combined High-Recall Blocking + Rich Similarity Features Experiment.
Amazon ML Challenge 2026 — Team BugHunters.

Combines:
1. Original high-recall Person 1 blocking strategies (93.43% recall ceiling).
2. Blocking meta-features (blocking_score, blocking_rank, blocking_score_ratio).
3. Rich linguistic similarity features (name_token_sort_ratio, 4gram Jaccard, IDF weighting).
4. Address consistency features (numeric conflict, building match, locality match, missing address flags).
5. Comprehensive threshold sweeping (0.50 - 0.95).
"""

import os
import sys
import time
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

from benchmark_validation_harness import build_validation_dataset
from dataset.person1.normalization import (
    normalize_name,
    clean_core_name,
    normalize_address,
    normalize_country,
)
from dataset.person1.blocking import BlockingIndex, generate_blocking_keys
from dataset.person2.features import ALL_FEATURES, preprocess_entity_record
from dataset.person3.metrics import evaluate_entity_resolution


COMBINED_FEATURES = ALL_FEATURES + [
    "name_core_exact",
    "name_token_sort_ratio",
    "name_char_4gram_jaccard",
    "name_first_token_match",
    "numeric_token_overlap",
    "numeric_token_conflict",
    "has_both_addresses",
    "either_address_empty",
    "blocking_score",
    "blocking_rank",
    "blocking_score_ratio",
]


def extract_numeric_tokens(addr: str) -> Set[str]:
    if not addr:
        return set()
    nums = set()
    for tok in addr.split():
        clean = tok.strip("#,.-/ ")
        if clean.isdigit():
            nums.add(clean)
        elif ("-" in tok or "/" in tok) and any(c.isdigit() for c in tok):
            nums.add(clean.replace("-", "").replace("/", ""))
    return nums


def compute_combined_pair_features(
    s1: Dict[str, object],
    cand: Dict[str, object],
    cand_source: str,
    blocking_score: float = 1.0,
    blocking_rank: float = 1.0,
    blocking_score_ratio: float = 1.0,
) -> Dict[str, float]:
    if not s1 or not cand:
        return {feat: 0.0 for feat in COMBINED_FEATURES}

    raw_n1, raw_n2 = s1["raw_name"], cand["raw_name"]
    norm_n1, norm_n2 = s1["norm_name"], cand["norm_name"]
    core_n1, core_n2 = s1["core_name"], cand["core_name"]

    name_exact = 1.0 if raw_n1 == raw_n2 and raw_n1 else 0.0
    name_norm_exact = 1.0 if norm_n1 == norm_n2 and norm_n1 else 0.0
    name_core_exact = 1.0 if core_n1 == core_n2 and core_n1 else 0.0

    name_lev = Levenshtein.normalized_similarity(norm_n1, norm_n2) if (norm_n1 and norm_n2) else 0.0

    t_set1, t_set2 = s1["core_token_set"], cand["core_token_set"]
    name_common = len(t_set1.intersection(t_set2))
    name_union = len(t_set1.union(t_set2))
    name_jaccard = (name_common / name_union) if name_union > 0 else 0.0
    min_tokens = min(len(t_set1), len(t_set2))
    name_token_overlap = (name_common / min_tokens) if min_tokens > 0 else 0.0

    name_set_ratio = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0 if (norm_n1 and norm_n2) else 0.0
    name_sort_ratio = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0 if (norm_n1 and norm_n2) else 0.0

    g1, g2 = s1["name_char_3grams"], cand["name_char_3grams"]
    name_gram_jaccard = (len(g1.intersection(g2)) / len(g1.union(g2))) if (g1 and g2) else 0.0

    from dataset.person2.enhanced_features import extract_char_ngrams
    g1_4 = extract_char_ngrams(norm_n1, 4)
    g2_4 = extract_char_ngrams(norm_n2, 4)
    name_4gram_jaccard = (len(g1_4.intersection(g2_4)) / len(g1_4.union(g2_4))) if (g1_4 and g2_4) else 0.0

    name_len_diff = abs(len(norm_n1) - len(norm_n2))
    name_tok_diff = abs(len(s1["name_tokens"]) - len(cand["name_tokens"]))
    name_contains = 1.0 if (norm_n1 in norm_n2 or norm_n2 in norm_n1) and (norm_n1 and norm_n2) else 0.0

    t1_first = s1["name_tokens"][0] if s1["name_tokens"] else ""
    t2_first = cand["name_tokens"][0] if cand["name_tokens"] else ""
    name_first_match = 1.0 if t1_first == t2_first and t1_first else 0.0

    # Address
    raw_a1, raw_a2 = s1["raw_addr"], cand["raw_addr"]
    norm_a1, norm_a2 = s1["norm_addr"], cand["norm_addr"]

    has_both_addr = 1.0 if (norm_a1 and norm_a2) else 0.0
    either_empty = 1.0 if (not norm_a1 or not norm_a2) else 0.0

    addr_exact = 1.0 if raw_a1 == raw_a2 and raw_a1 else 0.0
    addr_norm_exact = 1.0 if norm_a1 == norm_a2 and norm_a1 else 0.0
    addr_lev = Levenshtein.normalized_similarity(norm_a1, norm_a2) if (norm_a1 and norm_a2) else 0.0

    at_set1, at_set2 = s1["addr_token_set"], cand["addr_token_set"]
    addr_common = len(at_set1.intersection(at_set2))
    addr_union = len(at_set1.union(at_set2))
    addr_jaccard = (addr_common / addr_union) if addr_union > 0 else 0.0
    min_atoks = min(len(at_set1), len(at_set2))
    addr_overlap = (addr_common / min_atoks) if min_atoks > 0 else 0.0

    addr_set_ratio = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0 if (norm_a1 and norm_a2) else 0.0
    ag1, ag2 = s1["addr_char_3grams"], cand["addr_char_3grams"]
    addr_gram_jaccard = (len(ag1.intersection(ag2)) / len(ag1.union(ag2))) if (ag1 and ag2) else 0.0

    addr_len_diff = abs(len(norm_a1) - len(norm_a2))
    addr_tok_diff = abs(len(s1["addr_tokens"]) - len(cand["addr_tokens"]))
    addr_contains = 1.0 if (norm_a1 in norm_a2 or norm_a2 in norm_a1) and (norm_a1 and norm_a2) else 0.0

    b1, b2 = s1["bldg_no"], cand["bldg_no"]
    bldg_match = 0.5 if (not b1 or not b2) else (1.0 if b1 == b2 else 0.0)

    loc1, loc2 = s1["locality_set"], cand["locality_set"]
    if not loc1 or not loc2:
        loc_match = 0.5
    else:
        inter = len(loc1.intersection(loc2))
        loc_match = 1.0 if inter >= 2 else (0.75 if inter == 1 else 0.0)

    nums1 = extract_numeric_tokens(norm_a1)
    nums2 = extract_numeric_tokens(norm_a2)
    if nums1 and nums2:
        num_inter = len(nums1.intersection(nums2))
        num_overlap = float(num_inter)
        num_conflict = 1.0 if num_inter == 0 else 0.0
    else:
        num_overlap = 0.0
        num_conflict = 0.0

    c1, c2 = s1["country"], cand["country"]
    country_match = 1.0 if c1 == c2 and c1 else 0.0
    cand_src_is_s2 = 1.0 if cand_source == "source2" else 0.0

    return {
        "name_exact": name_exact,
        "name_normalized_exact": name_norm_exact,
        "name_levenshtein": name_lev,
        "name_jaccard": name_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_token_set_ratio": name_set_ratio,
        "name_char_3gram_jaccard": name_gram_jaccard,
        "name_length_difference": float(name_len_diff),
        "name_token_count_difference": float(name_tok_diff),
        "common_name_tokens": float(name_common),
        "name_contains_other": name_contains,
        "address_exact": addr_exact,
        "address_normalized_exact": addr_norm_exact,
        "address_levenshtein": addr_lev,
        "address_jaccard": addr_jaccard,
        "address_token_overlap": addr_overlap,
        "address_token_set_ratio": addr_set_ratio,
        "address_char_3gram_jaccard": addr_gram_jaccard,
        "address_length_difference": float(addr_len_diff),
        "address_token_count_difference": float(addr_tok_diff),
        "common_address_tokens": float(addr_common),
        "address_contains_other": addr_contains,
        "building_number_match": bldg_match,
        "locality_match": loc_match,
        "country_match": country_match,
        "candidate_source_is_s2": cand_src_is_s2,
        "name_core_exact": name_core_exact,
        "name_token_sort_ratio": name_sort_ratio,
        "name_char_4gram_jaccard": name_4gram_jaccard,
        "name_first_token_match": name_first_match,
        "numeric_token_overlap": num_overlap,
        "numeric_token_conflict": num_conflict,
        "has_both_addresses": has_both_addr,
        "either_address_empty": either_empty,
        "blocking_score": float(blocking_score),
        "blocking_rank": float(blocking_rank),
        "blocking_score_ratio": float(blocking_score_ratio),
    }


def run_experiment():
    print("=" * 90, flush=True)
    print("RUNNING COMBINED OPTIMAL PIPELINE EXPERIMENT", flush=True)
    print("=" * 90, flush=True)

    random_seed = 42
    val_ratio = 0.20

    s1_all, s2_all, s3_all, gt_map, gt_pairs = build_validation_dataset(random_seed=random_seed)

    for df in [s1_all, s2_all, s3_all]:
        df["norm_name"] = df["business_name"].apply(normalize_name)
        df["core_name"] = df["norm_name"].apply(clean_core_name)
        df["norm_addr"] = df["business_address"].apply(normalize_address)
        df["norm_country"] = df["country"].apply(normalize_country)

    np.random.seed(random_seed)
    all_s1_ids = list(s1_all["entity_id"].values)
    np.random.shuffle(all_s1_ids)

    n_val = int(len(all_s1_ids) * val_ratio)
    val_s1_ids = set(all_s1_ids[:n_val])
    train_s1_ids = set(all_s1_ids[n_val:])

    s1_train_df = s1_all[s1_all["entity_id"].isin(train_s1_ids)].copy().reset_index(drop=True)
    s1_val_df = s1_all[s1_all["entity_id"].isin(val_s1_ids)].copy().reset_index(drop=True)

    # Inverted Index with Original High-Recall Blocking Strategies
    index = BlockingIndex(max_block_size=1000)
    index.add_candidates(s2_all, source_name="source2")
    index.add_candidates(s3_all, source_name="source3")
    index.prune_large_blocks()

    def get_cand_features(s1_df):
        s1_keys_list = generate_blocking_keys(s1_df)
        records = []
        cands_per_s1 = {}
        for s1_id, keys in zip(s1_df["entity_id"], s1_keys_list):
            cand_scores = defaultdict(int)
            cand_src_map = {}
            for k in keys:
                if k in index.index:
                    for cid, src in index.index[k]:
                        cand_scores[cid] += 1
                        cand_src_map[cid] = src
            sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))[:300]
            max_s = sorted_cands[0][1] if sorted_cands else 1.0

            c_list = []
            for rank, (cid, score) in enumerate(sorted_cands, 1):
                records.append({
                    "source1_id": s1_id,
                    "candidate_id": cid,
                    "candidate_source": cand_src_map[cid],
                    "blocking_score": float(score),
                    "blocking_rank": float(rank),
                    "blocking_score_ratio": float(score / max_s),
                })
                c_list.append(cid)
            cands_per_s1[s1_id] = c_list
        return pd.DataFrame(records), cands_per_s1

    train_cands_df, train_cands_map = get_cand_features(s1_train_df)
    val_cands_df, val_cands_map = get_cand_features(s1_val_df)

    val_gt_pairs = {p for p in gt_pairs if p[0] in val_s1_ids}
    val_cand_pairs_set = set(zip(val_cands_df["source1_id"], val_cands_df["candidate_id"]))
    blocking_hits = len(val_gt_pairs.intersection(val_cand_pairs_set))
    cand_recall_ceiling = blocking_hits / max(1, len(val_gt_pairs))

    val_cand_lens = np.array([len(val_cands_map[s1]) for s1 in val_s1_ids])
    avg_cands = float(np.mean(val_cand_lens))

    print(f"Candidate Recall Ceiling: {cand_recall_ceiling*100:.2f}% | Avg candidates/S1: {avg_cands:.1f}", flush=True)

    # Feature extraction
    needed_s1 = set(train_cands_df["source1_id"]).union(set(val_cands_df["source1_id"]))
    needed_cands = set(train_cands_df["candidate_id"]).union(set(val_cands_df["candidate_id"]))

    from dataset.person2.features import load_entity_lookups
    lookups = load_entity_lookups(s1_all, s2_all, s3_all, needed_s1, needed_cands)

    def extract_matrix(cand_df, has_gt=True):
        rows = []
        labels = []
        for row in cand_df.itertuples(index=False):
            s1_rec = lookups.get(row.source1_id)
            c_rec = lookups.get(row.candidate_id)
            d = compute_combined_pair_features(
                s1=s1_rec,
                cand=c_rec,
                cand_source=row.candidate_source,
                blocking_score=row.blocking_score,
                blocking_rank=row.blocking_rank,
                blocking_score_ratio=row.blocking_score_ratio,
            )
            rows.append(d)
            if has_gt:
                labels.append(1 if (row.source1_id, row.candidate_id) in gt_pairs else 0)
        df_feat = pd.DataFrame(rows)
        df_feat["source1_id"] = cand_df["source1_id"].values
        df_feat["candidate_id"] = cand_df["candidate_id"].values
        df_feat["candidate_source"] = cand_df["candidate_source"].values
        if has_gt:
            df_feat["label"] = labels
        return df_feat

    print("Extracting combined features...", flush=True)
    train_feat_df = extract_matrix(train_cands_df, has_gt=True)
    val_feat_df = extract_matrix(val_cands_df, has_gt=True)

    X_tr = train_feat_df[COMBINED_FEATURES].values
    y_tr = train_feat_df["label"].values
    X_va = val_feat_df[COMBINED_FEATURES].values
    y_va = val_feat_df["label"].values

    # Train LightGBM model
    print("Training LightGBM model on combined feature set...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=45,
        max_depth=9,
        min_child_samples=25,
        feature_fraction=0.85,
        subsample=0.85,
        random_state=random_seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)

    val_probs = model.predict_proba(X_va)[:, 1]
    val_feat_df["prob"] = val_probs

    # Threshold sweeping
    print("\n--- THRESHOLD SWEEP ---", flush=True)
    best_t = 0.80
    best_f05 = 0.0
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.95]:
        preds_map = defaultdict(set)
        passing = val_feat_df[val_feat_df["prob"] >= t]
        for s1, cid in zip(passing["source1_id"], passing["candidate_id"]):
            if cid.startswith(("S2-", "S3-")):
                preds_map[s1].add(cid)

        pred_dict = {s: preds_map.get(s, set()) for s in val_s1_ids}
        m = evaluate_entity_resolution(gt_map, pred_dict, evaluated_s1_ids=list(val_s1_ids), beta=0.5)
        star = " *" if m["macro_f05"] > best_f05 else ""
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_t = t
        print(f"  T = {t:.2f}: Macro F0.5 = {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} | SingAcc: {m['singleton_accuracy']:.4f}{star}", flush=True)

    # Evaluate best threshold
    best_passing = val_feat_df[val_feat_df["prob"] >= best_t]
    best_preds_map = defaultdict(set)
    for s1, cid in zip(best_passing["source1_id"], best_passing["candidate_id"]):
        if cid.startswith(("S2-", "S3-")):
            best_preds_map[s1].add(cid)

    best_pred_dict = {s: best_preds_map.get(s, set()) for s in val_s1_ids}
    best_metrics = evaluate_entity_resolution(gt_map, best_pred_dict, evaluated_s1_ids=list(val_s1_ids), beta=0.5)

    country_map = dict(zip(s1_all["entity_id"], s1_all["country"]))
    us_val = [s for s in val_s1_ids if country_map.get(s) == "US"]
    india_val = [s for s in val_s1_ids if country_map.get(s) == "India"]

    us_best = evaluate_entity_resolution(gt_map, best_pred_dict, evaluated_s1_ids=us_val, beta=0.5) if us_val else {}
    india_best = evaluate_entity_resolution(gt_map, best_pred_dict, evaluated_s1_ids=india_val, beta=0.5) if india_val else {}

    print("\n" + "=" * 90, flush=True)
    print("FINAL COMBINED PIPELINE PERFORMANCE:", flush=True)
    print(f"  Optimal Threshold:      T = {best_t:.2f}")
    print(f"  Macro F0.5:             {best_metrics['macro_f05']:.4f} (Baseline: 0.8592, Delta: +{best_metrics['macro_f05']-0.8592:.4f})")
    print(f"  Macro Precision:        {best_metrics['macro_precision']:.4f} (Baseline: 0.9074)")
    print(f"  Macro Recall:           {best_metrics['macro_recall']:.4f} (Baseline: 0.8175)")
    print(f"  Singleton Accuracy:     {best_metrics['singleton_accuracy']:.4f} (Baseline: 0.0962)")
    print(f"  Non-singleton F0.5:     {best_metrics['non_singleton_macro_f05']:.4f} (Baseline: 0.9037)")
    print(f"  Candidate Recall Ceiling:{cand_recall_ceiling*100:.2f}% (Baseline: 93.43%)")
    print(f"  Average Candidates / S1: {avg_cands:.1f} (Baseline: 187.8)")
    print(f"  US Macro F0.5:          {us_best.get('macro_f05', 0):.4f} (Baseline: 0.8961)")
    print(f"  India Macro F0.5:       {india_best.get('macro_f05', 0):.4f} (Baseline: 0.8069)")
    print("=" * 90, flush=True)

    # Save artifact
    model_save_path = os.path.join(dataset_dir, "person2", "models", "best_combined_model.pkl")
    import joblib
    joblib.dump({
        "model": model,
        "feature_names": COMBINED_FEATURES,
        "optimal_threshold": best_t,
        "metrics": best_metrics,
        "cand_recall_ceiling": cand_recall_ceiling,
    }, model_save_path)
    print(f"Saved model to: {model_save_path}", flush=True)


if __name__ == "__main__":
    run_experiment()
