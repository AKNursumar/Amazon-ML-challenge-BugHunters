"""
Final Production Test Inference Runner for Amazon ML Challenge 2026.
Team: BugHunters

Applies the best validated entity resolution pipeline:
- Person 1 multi-strategy blocking (NP4, NP6, TOK, TOK3, NSORT, NCOMP, ADDR, UNIT, ADDR_LOC).
- 37 combined linguistic, address, numeric, and candidate meta-features.
- Champion LightGBM classifier with calibrated country-specific thresholds (US: 0.55, India: 0.50, France: 0.55).
- Generates official submission files:
  * output/matching_results.tsv
  * output/candidate_pairs.tsv
- Runs official validator (utils/validate_submission.py).
"""

import os
import sys
import gc
import time
import joblib
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(cur_dir, "dataset")
for p in [cur_dir, dataset_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from test_combined_pipeline import COMBINED_FEATURES, compute_combined_pair_features
from dataset.person1.normalization import (
    normalize_name,
    clean_core_name,
    normalize_address,
    normalize_country,
)
from dataset.person1.blocking import BlockingIndex, generate_blocking_keys
from dataset.person2.features import load_entity_lookups
from dataset.person3.decision import format_submission_files
from utils.validate_submission import validate


def run_production_inference(
    test_dir: str = "dataset/test",
    model_path: str = "dataset/person2/models/best_combined_model.pkl",
    output_matching_path: str = "output/matching_results.tsv",
    output_candidate_path: str = "output/candidate_pairs.tsv",
    threshold_us: float = 0.55,
    threshold_india: float = 0.50,
    threshold_france: float = 0.55,
    max_block_size: int = 1000,
    max_candidates_per_query: int = 300,
    s1_batch_size: int = 50_000,
):
    start_time = time.time()
    print("=" * 85, flush=True)
    print("RUNNING FINAL TEST SET INFERENCE PIPELINE (BugHunters Champion Model)", flush=True)
    print(f"Test Directory:      {test_dir}", flush=True)
    print(f"Model Path:          {model_path}", flush=True)
    print(f"Thresholds:          US={threshold_us:.2f}, India={threshold_india:.2f}, France={threshold_france:.2f}", flush=True)
    print(f"Matching Results:    {output_matching_path}", flush=True)
    print(f"Candidate Pairs:     {output_candidate_path}", flush=True)
    print("=" * 85, flush=True)

    # 1. Load trained Champion LightGBM Model
    print("\n[Step 1/6] Loading Champion LightGBM Model...", flush=True)
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload.get("feature_names", COMBINED_FEATURES)
    assert feature_names == COMBINED_FEATURES, "Feature names mismatch!"
    print(f"  Model loaded: {type(model).__name__} ({len(feature_names)} features)", flush=True)

    # 2. Read Source 1 IDs in exact order
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n[Step 2/6] Loading Source 1 records from {test_s1_path}...", flush=True)
    s1_full_df = pd.read_csv(test_s1_path, sep="\t", dtype=str).fillna("")
    all_required_s1_ids = s1_full_df["entity_id"].tolist()
    print(f"  Total Test Source 1 entities: {len(all_required_s1_ids):,}", flush=True)

    countries = s1_full_df["country"].unique().tolist()
    print(f"  Test countries present: {countries}", flush=True)

    matching_map: Dict[str, List[str]] = {}
    candidates_map: Dict[str, List[str]] = {}

    for s1_id in all_required_s1_ids:
        matching_map[s1_id] = []
        candidates_map[s1_id] = []

    total_cands_evaluated = 0
    total_matches_predicted = 0

    country_thresholds = {
        "US": threshold_us,
        "India": threshold_india,
        "France": threshold_france,
    }

    # 3. Process country by country
    for c_idx, country in enumerate(countries, 1):
        c_start = time.time()
        c_thresh = country_thresholds.get(country, 0.55)
        print("\n" + "-" * 85, flush=True)
        print(f"[{c_idx}/{len(countries)}] Processing Country: '{country}' (Decision Threshold T = {c_thresh:.2f})", flush=True)
        print("-" * 85, flush=True)

        s1_c = s1_full_df[s1_full_df["country"] == country].copy().reset_index(drop=True)
        print(f"  Source 1 entities for {country}: {len(s1_c):,}", flush=True)

        # Stream load S2 and S3 for this country
        print(f"  Loading Source 2 records for {country}...", flush=True)
        s2_c_chunks = []
        for chunk in pd.read_csv(test_s2_path, sep="\t", chunksize=500_000, dtype=str, keep_default_na=False):
            m = chunk["country"] == country
            if m.any():
                s2_c_chunks.append(chunk[m])
        s2_c = pd.concat(s2_c_chunks, ignore_index=True) if s2_c_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        del s2_c_chunks
        gc.collect()
        print(f"  Loaded {len(s2_c):,} Source 2 records.", flush=True)

        print(f"  Loading Source 3 records for {country}...", flush=True)
        s3_c_chunks = []
        for chunk in pd.read_csv(test_s3_path, sep="\t", chunksize=500_000, dtype=str, keep_default_na=False):
            m = chunk["country"] == country
            if m.any():
                s3_c_chunks.append(chunk[m])
        s3_c = pd.concat(s3_c_chunks, ignore_index=True) if s3_c_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        del s3_c_chunks
        gc.collect()
        print(f"  Loaded {len(s3_c):,} Source 3 records.", flush=True)

        # Normalization
        print("  Normalizing Source 2 & Source 3 fields...", flush=True)
        s2_c["norm_name"] = s2_c["business_name"].apply(normalize_name)
        s2_c["core_name"] = s2_c["norm_name"].apply(clean_core_name)
        s2_c["norm_addr"] = s2_c["business_address"].apply(normalize_address)
        s2_c["norm_country"] = s2_c["country"].apply(normalize_country)

        s3_c["norm_name"] = s3_c["business_name"].apply(normalize_name)
        s3_c["core_name"] = s3_c["norm_name"].apply(clean_core_name)
        s3_c["norm_addr"] = s3_c["business_address"].apply(normalize_address)
        s3_c["norm_country"] = s3_c["country"].apply(normalize_country)

        # Build Inverted Blocking Index
        print("  Building inverted blocking index...", flush=True)
        t_idx = time.time()
        index = BlockingIndex(max_block_size=max_block_size)
        index.add_candidates(s2_c, source_name="source2")
        index.add_candidates(s3_c, source_name="source3")
        pruned = index.prune_large_blocks()
        print(f"  Inverted index built ({len(index.index):,} keys, {pruned:,} pruned) in {time.time()-t_idx:.2f}s", flush=True)

        n_s1_country = len(s1_c)
        num_batches = (n_s1_country + s1_batch_size - 1) // s1_batch_size
        print(f"  Predicting across {num_batches} batches of {s1_batch_size:,} S1 entities...", flush=True)

        country_cands_count = 0
        country_matches_count = 0

        for b_idx in range(num_batches):
            b_start = b_idx * s1_batch_size
            b_end = min((b_idx + 1) * s1_batch_size, n_s1_country)
            s1_batch = s1_c.iloc[b_start:b_end].copy().reset_index(drop=True)

            t_batch = time.time()

            s1_batch["norm_name"] = s1_batch["business_name"].apply(normalize_name)
            s1_batch["core_name"] = s1_batch["norm_name"].apply(clean_core_name)
            s1_batch["norm_addr"] = s1_batch["business_address"].apply(normalize_address)
            s1_batch["norm_country"] = s1_batch["country"].apply(normalize_country)

            s1_keys_list = generate_blocking_keys(s1_batch)
            s1_ids = s1_batch["entity_id"].tolist()

            batch_pairs_s1 = []
            batch_pairs_cand = []
            batch_pairs_src = []
            batch_b_scores = []
            batch_b_ranks = []
            batch_b_ratios = []

            s1_candidates_dict = {}

            for s1_id, keys in zip(s1_ids, s1_keys_list):
                cand_scores = defaultdict(int)
                cand_src_map = {}

                for k in keys:
                    if k in index.index:
                        cand_list = index.index[k]
                        if len(cand_list) <= index.max_block_size:
                            for cand_id, cand_src in cand_list:
                                cand_scores[cand_id] += 1
                                cand_src_map[cand_id] = cand_src

                sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))
                if max_candidates_per_query and len(sorted_cands) > max_candidates_per_query:
                    sorted_cands = sorted_cands[:max_candidates_per_query]

                max_s = sorted_cands[0][1] if sorted_cands else 1.0

                c_id_list = []
                for rank, (cand_id, score) in enumerate(sorted_cands, 1):
                    c_id_list.append(cand_id)
                    batch_pairs_s1.append(s1_id)
                    batch_pairs_cand.append(cand_id)
                    batch_pairs_src.append(cand_src_map[cand_id])
                    batch_b_scores.append(float(score))
                    batch_b_ranks.append(float(rank))
                    batch_b_ratios.append(float(score / max_s))

                s1_candidates_dict[s1_id] = c_id_list

            num_pairs = len(batch_pairs_s1)
            country_cands_count += num_pairs

            if num_pairs > 0:
                unique_batch_s1 = set(batch_pairs_s1)
                unique_batch_cands = set(batch_pairs_cand)

                lookups = load_entity_lookups(
                    s1_df=s1_batch,
                    s2_df=s2_c,
                    s3_df=s3_c,
                    needed_s1_ids=unique_batch_s1,
                    needed_cand_ids=unique_batch_cands,
                )

                # Extract features
                feat_rows = []
                for i in range(num_pairs):
                    s1_rec = lookups.get(batch_pairs_s1[i])
                    c_rec = lookups.get(batch_pairs_cand[i])
                    d = compute_combined_pair_features(
                        s1=s1_rec,
                        cand=c_rec,
                        cand_source=batch_pairs_src[i],
                        blocking_score=batch_b_scores[i],
                        blocking_rank=batch_b_ranks[i],
                        blocking_score_ratio=batch_b_ratios[i],
                    )
                    feat_rows.append(d)

                feat_df = pd.DataFrame(feat_rows)
                del lookups, feat_rows
                gc.collect()

                X_batch = feat_df[feature_names].values
                probs = model.predict_proba(X_batch)[:, 1]

                # Aggregate matches above calibrated threshold
                s1_matches_dict = defaultdict(list)
                for s1_id, c_id, prob in zip(batch_pairs_s1, batch_pairs_cand, probs):
                    if prob >= c_thresh and c_id.startswith(("S2-", "S3-")):
                        s1_matches_dict[s1_id].append((c_id, float(prob)))

                del feat_df, X_batch, probs
                gc.collect()

                for s1_id in s1_ids:
                    c_list = s1_candidates_dict.get(s1_id, [])
                    candidates_map[s1_id] = c_list

                    m_tuples = s1_matches_dict.get(s1_id, [])
                    if m_tuples:
                        m_tuples.sort(key=lambda x: (-x[1], x[0]))
                        seen_m = set()
                        m_list = []
                        for cid, _ in m_tuples:
                            if cid not in seen_m:
                                seen_m.add(cid)
                                m_list.append(cid)
                        matching_map[s1_id] = m_list
                        country_matches_count += len(m_list)
                    else:
                        matching_map[s1_id] = []
            else:
                for s1_id in s1_ids:
                    candidates_map[s1_id] = []
                    matching_map[s1_id] = []

            b_time = time.time() - t_batch
            print(f"    Batch {b_idx+1}/{num_batches} ({len(s1_batch):,} S1s, {num_pairs:,} pairs) done in {b_time:.2f}s", flush=True)

            del s1_batch, s1_keys_list, batch_pairs_s1, batch_pairs_cand, batch_pairs_src
            del batch_b_scores, batch_b_ranks, batch_b_ratios, s1_candidates_dict
            gc.collect()

        total_cands_evaluated += country_cands_count
        total_matches_predicted += country_matches_count
        print(f"  Finished {country} in {time.time()-c_start:.2f}s | Candidates: {country_cands_count:,} | Matches: {country_matches_count:,}", flush=True)

        del index, s2_c, s3_c, s1_c
        gc.collect()

    del s1_full_df
    gc.collect()

    # 4. Format and write final submission files
    print("\n[Step 4/6] Formatting and writing submission files...", flush=True)
    n_rows, n_links = format_submission_files(
        matching_map=matching_map,
        candidates_map=candidates_map,
        required_s1_ids=all_required_s1_ids,
        matching_output_path=output_matching_path,
        candidate_output_path=output_candidate_path,
    )

    match_mb = os.path.getsize(output_matching_path) / (1024 * 1024)
    cand_mb = os.path.getsize(output_candidate_path) / (1024 * 1024)
    print(f"  Generated {output_matching_path} ({match_mb:.2f} MB, {n_rows:,} rows)", flush=True)
    print(f"  Generated {output_candidate_path} ({cand_mb:.2f} MB, {n_rows:,} rows)", flush=True)

    # 5. Calculate statistics
    print("\n[Step 5/6] Final prediction statistics...", flush=True)
    non_empty_count = sum(1 for m in matching_map.values() if len(m) > 0)
    empty_count = len(all_required_s1_ids) - non_empty_count
    avg_links = n_links / len(all_required_s1_ids)

    cand_lengths = [len(c) for c in candidates_map.values()]
    avg_cands = float(np.mean(cand_lengths))
    max_cands = int(np.max(cand_lengths))

    print(f"  Total S1 Entities:     {len(all_required_s1_ids):,}", flush=True)
    print(f"  Non-empty Predictions: {non_empty_count:,} ({non_empty_count/len(all_required_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Empty Predictions:     {empty_count:,} ({empty_count/len(all_required_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Total Match Links:     {n_links:,}", flush=True)
    print(f"  Average Links / S1:    {avg_links:.3f}", flush=True)
    print(f"  Average Candidates/S1: {avg_cands:.1f}", flush=True)
    print(f"  Max Candidates / S1:   {max_cands}", flush=True)

    # 6. Run Official Submission Validator
    print("\n[Step 6/6] Executing official submission validator...", flush=True)
    errors, warnings = validate(
        matching_path=output_matching_path,
        candidate_path=output_candidate_path,
        test_dir=test_dir,
        check_ids=False,
    )

    if errors:
        val_status = f"FAIL — {len(errors)} error(s)"
        for i, err in enumerate(errors, 1):
            print(f"  ERROR {i}: {err}", flush=True)
    else:
        val_status = "PASS — no blocking issues found. Safe to submit."
        print(f"  {val_status}", flush=True)

    total_time = time.time() - start_time
    print("=" * 85, flush=True)
    print(f"TEST INFERENCE COMPLETED IN {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print("=" * 85, flush=True)

    return {
        "test_s1_count": len(all_required_s1_ids),
        "non_empty": non_empty_count,
        "empty": empty_count,
        "total_links": n_links,
        "avg_links": avg_links,
        "avg_cands": avg_cands,
        "max_cands": max_cands,
        "thresholds": country_thresholds,
        "validator_result": val_status,
        "runtime_sec": total_time,
    }


if __name__ == "__main__":
    run_production_inference()
