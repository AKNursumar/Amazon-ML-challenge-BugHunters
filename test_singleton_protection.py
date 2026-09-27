"""
Singleton Protection & Calibrated Decision Rule Exploration.
Amazon ML Challenge 2026 — Team BugHunters.

Tests data-driven singleton protection / abstention rules on the validation set:
1. Max Probability Gate: If max_prob for S1 < T_max_prob, abstain (predict empty).
2. Name Evidence Gate: If candidate has name_lev < min_lev AND name_jaccard < min_jacc, require higher threshold.
3. Country-specific thresholds (US vs India).
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import joblib
import lightgbm as lgb

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(cur_dir, "dataset")
for p in [cur_dir, dataset_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from benchmark_validation_harness import build_validation_dataset
from test_combined_pipeline import COMBINED_FEATURES, compute_combined_pair_features
from dataset.person1.normalization import normalize_name, clean_core_name, normalize_address, normalize_country
from dataset.person1.blocking import BlockingIndex, generate_blocking_keys
from dataset.person2.features import load_entity_lookups
from dataset.person3.metrics import evaluate_entity_resolution


def run_singleton_experiments():
    print("=" * 85)
    print("TESTING SINGLETON PROTECTION & DECISION THRESHOLDING")
    print("=" * 85)

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

    index = BlockingIndex(max_block_size=1000)
    index.add_candidates(s2_all, source_name="source2")
    index.add_candidates(s3_all, source_name="source3")
    index.prune_large_blocks()

    def get_cands(s1_df):
        keys_list = generate_blocking_keys(s1_df)
        recs = []
        for s1_id, keys in zip(s1_df["entity_id"], keys_list):
            scores = defaultdict(int)
            src_map = {}
            for k in keys:
                if k in index.index:
                    for cid, src in index.index[k]:
                        scores[cid] += 1
                        src_map[cid] = src
            sorted_cands = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:300]
            max_s = sorted_cands[0][1] if sorted_cands else 1.0
            for rank, (cid, score) in enumerate(sorted_cands, 1):
                recs.append({
                    "source1_id": s1_id,
                    "candidate_id": cid,
                    "candidate_source": src_map[cid],
                    "blocking_score": float(score),
                    "blocking_rank": float(rank),
                    "blocking_score_ratio": float(score / max_s),
                })
        return pd.DataFrame(recs)

    train_cands = get_cands(s1_train_df)
    val_cands = get_cands(s1_val_df)

    needed_s1 = set(train_cands["source1_id"]).union(set(val_cands["source1_id"]))
    needed_cands = set(train_cands["candidate_id"]).union(set(val_cands["candidate_id"]))
    lookups = load_entity_lookups(s1_all, s2_all, s3_all, needed_s1, needed_cands)

    def extract_feat(cand_df, has_gt=True):
        rows = []
        labels = []
        for r in cand_df.itertuples(index=False):
            s1_r = lookups.get(r.source1_id)
            c_r = lookups.get(r.candidate_id)
            d = compute_combined_pair_features(s1_r, c_r, r.candidate_source, r.blocking_score, r.blocking_rank, r.blocking_score_ratio)
            rows.append(d)
            if has_gt:
                labels.append(1 if (r.source1_id, r.candidate_id) in gt_pairs else 0)
        df_feat = pd.DataFrame(rows)
        df_feat["source1_id"] = cand_df["source1_id"].values
        df_feat["candidate_id"] = cand_df["candidate_id"].values
        df_feat["candidate_source"] = cand_df["candidate_source"].values
        if has_gt:
            df_feat["label"] = labels
        return df_feat

    train_feat = extract_feat(train_cands, has_gt=True)
    val_feat = extract_feat(val_cands, has_gt=True)

    X_tr = train_feat[COMBINED_FEATURES].values
    y_tr = train_feat["label"].values
    X_va = val_feat[COMBINED_FEATURES].values

    # Load / Train model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=45,
        max_depth=9,
        min_child_samples=25,
        feature_fraction=0.85,
        subsample=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    val_feat["prob"] = model.predict_proba(X_va)[:, 1]

    # Grid search on:
    # 1. Base threshold T_base (0.50 - 0.80)
    # 2. Max probability gate T_max_gate (e.g. S1 must have at least one cand with prob >= T_max_gate)
    # 3. Minimum name similarity guard (if name_lev < 0.50, require higher threshold)

    print("\n--- GRID SEARCH: Base Threshold + Max Probability Gate ---")
    results = []
    best_f05 = 0.0
    best_config = None

    for t_base in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for t_gate in [t_base, t_base + 0.05, t_base + 0.10, t_base + 0.15, t_base + 0.20]:
            preds_map = defaultdict(set)
            # Group predictions by S1
            s1_cands = defaultdict(list)
            for s1_id, cid, prob, n_lev in zip(val_feat["source1_id"], val_feat["candidate_id"], val_feat["prob"], val_feat["name_levenshtein"]):
                if prob >= t_base and cid.startswith(("S2-", "S3-")):
                    s1_cands[s1_id].append((cid, prob, n_lev))

            for s1_id, cand_list in s1_cands.items():
                if cand_list:
                    max_p = max(c[1] for c in cand_list)
                    # Gate condition: at least one candidate must have high confidence
                    if max_p >= t_gate:
                        # Retain passing candidates
                        for cid, prob, n_lev in cand_list:
                            # If weak name similarity (< 0.55), require prob >= t_gate
                            if n_lev < 0.55 and prob < t_gate:
                                continue
                            preds_map[s1_id].add(cid)

            pred_dict = {s: preds_map.get(s, set()) for s in val_s1_ids}
            m = evaluate_entity_resolution(gt_map, pred_dict, evaluated_s1_ids=list(val_s1_ids), beta=0.5)

            results.append({
                "t_base": t_base,
                "t_gate": t_gate,
                "macro_f05": m["macro_f05"],
                "prec": m["macro_precision"],
                "rec": m["macro_recall"],
                "sing_acc": m["singleton_accuracy"],
                "non_sing_f05": m["non_singleton_macro_f05"],
            })

            star = " *" if m["macro_f05"] > best_f05 else ""
            if m["macro_f05"] > best_f05:
                best_f05 = m["macro_f05"]
                best_config = (t_base, t_gate, m)

            print(f"  T_base={t_base:.2f}, T_gate={t_gate:.2f} -> Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} | SingAcc: {m['singleton_accuracy']:.4f}{star}")

    print("\n" + "=" * 85)
    print(f"BEST CONFIGURATION: T_base = {best_config[0]:.2f}, T_gate = {best_config[1]:.2f}")
    print(f"  Macro F0.5:         {best_config[2]['macro_f05']:.4f} (Baseline: 0.8592, Delta: +{best_config[2]['macro_f05']-0.8592:.4f})")
    print(f"  Macro Precision:    {best_config[2]['macro_precision']:.4f}")
    print(f"  Macro Recall:       {best_config[2]['macro_recall']:.4f}")
    print(f"  Singleton Accuracy: {best_config[2]['singleton_accuracy']:.4f}")
    print("=" * 85)

    return best_config


if __name__ == "__main__":
    run_singleton_experiments()
