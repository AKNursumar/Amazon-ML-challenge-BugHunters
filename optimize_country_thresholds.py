"""
Country-Specific Threshold Optimization Script.
Amazon ML Challenge 2026 — Team BugHunters.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import joblib

from benchmark_validation_harness import build_validation_dataset
from test_combined_pipeline import COMBINED_FEATURES, compute_combined_pair_features
from dataset.person1.normalization import normalize_name, clean_core_name, normalize_address, normalize_country
from dataset.person1.blocking import BlockingIndex, generate_blocking_keys
from dataset.person2.features import load_entity_lookups
from dataset.person3.metrics import evaluate_entity_resolution

s1_all, s2_all, s3_all, gt_map, gt_pairs = build_validation_dataset(random_seed=42)

for df in [s1_all, s2_all, s3_all]:
    df['norm_name'] = df['business_name'].apply(normalize_name)
    df['core_name'] = df['norm_name'].apply(clean_core_name)
    df['norm_addr'] = df['business_address'].apply(normalize_address)
    df['norm_country'] = df['country'].apply(normalize_country)

all_s1_ids = list(s1_all['entity_id'].values)
np.random.seed(42)
np.random.shuffle(all_s1_ids)

n_val = int(len(all_s1_ids) * 0.20)
val_s1_ids = set(all_s1_ids[:n_val])

payload = joblib.load('dataset/person2/models/best_combined_model.pkl')
model = payload['model']

index = BlockingIndex(max_block_size=1000)
index.add_candidates(s2_all, source_name='source2')
index.add_candidates(s3_all, source_name='source3')
index.prune_large_blocks()

s1_val_df = s1_all[s1_all['entity_id'].isin(val_s1_ids)].copy().reset_index(drop=True)

s1_keys_list = generate_blocking_keys(s1_val_df)
records = []
for s1_id, keys in zip(s1_val_df['entity_id'], s1_keys_list):
    cand_scores = defaultdict(int)
    cand_src_map = {}
    for k in keys:
        if k in index.index:
            for cid, src in index.index[k]:
                cand_scores[cid] += 1
                cand_src_map[cid] = src
    sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))[:300]
    max_s = sorted_cands[0][1] if sorted_cands else 1.0
    for rank, (cid, score) in enumerate(sorted_cands, 1):
        records.append({
            'source1_id': s1_id,
            'candidate_id': cid,
            'candidate_source': cand_src_map[cid],
            'blocking_score': float(score),
            'blocking_rank': float(rank),
            'blocking_score_ratio': float(score / max_s),
        })

val_cands_df = pd.DataFrame(records)
needed_s1 = set(val_cands_df['source1_id'])
needed_cands = set(val_cands_df['candidate_id'])
lookups = load_entity_lookups(s1_all, s2_all, s3_all, needed_s1, needed_cands)

feat_rows = []
for r in val_cands_df.itertuples(index=False):
    s1_r = lookups.get(r.source1_id)
    c_r = lookups.get(r.candidate_id)
    d = compute_combined_pair_features(s1_r, c_r, r.candidate_source, r.blocking_score, r.blocking_rank, r.blocking_score_ratio)
    feat_rows.append(d)

val_feat_df = pd.DataFrame(feat_rows)
val_feat_df['source1_id'] = val_cands_df['source1_id'].values
val_feat_df['candidate_id'] = val_cands_df['candidate_id'].values
val_feat_df['candidate_source'] = val_cands_df['candidate_source'].values
val_feat_df['prob'] = model.predict_proba(val_feat_df[COMBINED_FEATURES].values)[:, 1]

country_map = dict(zip(s1_all['entity_id'], s1_all['country']))
val_feat_df['country'] = val_feat_df['source1_id'].map(country_map)

print('--- PER-COUNTRY THRESHOLD GRID SEARCH ---')
best_f05 = 0.0
best_t_us = 0.55
best_t_in = 0.55

for t_us in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    for t_in in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds_map = defaultdict(set)
        for s1_id, cid, prob, ctry in zip(val_feat_df['source1_id'], val_feat_df['candidate_id'], val_feat_df['prob'], val_feat_df['country']):
            thresh = t_us if ctry == 'US' else t_in
            if prob >= thresh and cid.startswith(('S2-', 'S3-')):
                preds_map[s1_id].add(cid)

        pred_dict = {s: preds_map.get(s, set()) for s in val_s1_ids}
        m = evaluate_entity_resolution(gt_map, pred_dict, evaluated_s1_ids=list(val_s1_ids), beta=0.5)

        if m['macro_f05'] > best_f05:
            best_f05 = m['macro_f05']
            best_t_us = t_us
            best_t_in = t_in
            print(f"New Best: T_US={t_us:.2f}, T_India={t_in:.2f} -> Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f}")

print(f"\nOPTIMAL PER-COUNTRY CONFIG: T_US={best_t_us:.2f}, T_India={best_t_in:.2f} -> Macro F0.5: {best_f05:.4f}")
