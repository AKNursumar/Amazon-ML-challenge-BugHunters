"""
Full Test Set Inference Pipeline for Amazon ML Challenge 2026.
Team: BugHunters

Applies the exact Person 1 blocking, Person 2 feature pipeline, and trained LightGBM model (threshold=0.80)
to generate the official test submission files:
- output/matching_results.tsv
- output/candidate_pairs.tsv
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

# Ensure correct sys.path
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
from dataset.person3.decision import format_submission_files
from utils.validate_submission import validate


def run_full_test_inference(
    test_dir: str = "dataset/test",
    model_path: str = "dataset/person2/models/best_model.pkl",
    output_matching_path: str = "output/matching_results.tsv",
    output_candidate_path: str = "output/candidate_pairs.tsv",
    threshold: float = 0.80,
    max_block_size: int = 1000,
    max_candidates_per_query: int = 300,
    s1_batch_size: int = 50_000,
):
    start_time = time.time()
    print("=" * 85, flush=True)
    print("RUNNING OFFICIAL TEST SET INFERENCE PIPELINE", flush=True)
    print(f"Test Directory: {test_dir}", flush=True)
    print(f"Model Path:     {model_path}", flush=True)
    print(f"Threshold:      {threshold:.2f}", flush=True)
    print(f"Matching Out:   {output_matching_path}", flush=True)
    print(f"Candidate Out:  {output_candidate_path}", flush=True)
    print("=" * 85, flush=True)

    # 1. Load trained LightGBM model
    print("\n[Step 1/6] Loading LightGBM champion model...", flush=True)
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload.get("feature_names", ALL_FEATURES)
    assert feature_names == ALL_FEATURES, "Feature names mismatch!"
    print(f"  Model successfully loaded: {payload.get('model_name', type(model).__name__)}", flush=True)
    print(f"  Using threshold: {threshold:.2f}", flush=True)

    # 2. Read full list of required Source 1 IDs in exact order
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n[Step 2/6] Loading Source 1 records from {test_s1_path}...", flush=True)
    s1_full_df = pd.read_csv(test_s1_path, sep="\t", dtype=str).fillna("")
    all_required_s1_ids = s1_full_df["entity_id"].tolist()
    print(f"  Total Test Source 1 entities: {len(all_required_s1_ids):,}", flush=True)

    countries = s1_full_df["country"].unique().tolist()
    print(f"  Test countries: {countries}", flush=True)

    # Dictionaries to store final outputs
    matching_map: Dict[str, List[str]] = {}
    candidates_map: Dict[str, List[str]] = {}

    # Initialize all with empty lists
    for s1_id in all_required_s1_ids:
        matching_map[s1_id] = []
        candidates_map[s1_id] = []

    total_candidates_evaluated = 0
    total_matches_found = 0

    # 3. Process country by country to minimize memory usage
    for c_idx, country in enumerate(countries, 1):
        c_start = time.time()
        print("\n" + "-" * 85, flush=True)
        print(f"[{c_idx}/{len(countries)}] Processing Country: '{country}'", flush=True)
        print("-" * 85, flush=True)

        s1_c = s1_full_df[s1_full_df["country"] == country].copy().reset_index(drop=True)
        print(f"  Source 1 records for {country}: {len(s1_c):,}", flush=True)

        # Stream load Source 2 and Source 3 for this country
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

        # Build Blocking Inverted Index
        print("  Building inverted blocking index...", flush=True)
        t_idx = time.time()
        index = BlockingIndex(max_block_size=max_block_size)
        index.add_candidates(s2_c, source_name="source2")
        index.add_candidates(s3_c, source_name="source3")
        pruned = index.prune_large_blocks()
        print(f"  Inverted index built ({len(index.index):,} keys, {pruned:,} pruned) in {time.time()-t_idx:.2f}s", flush=True)

        # Query index and predict for S1 in batches
        n_s1_country = len(s1_c)
        num_batches = (n_s1_country + s1_batch_size - 1) // s1_batch_size
        print(f"  Querying candidates and predicting for {n_s1_country:,} S1 entities across {num_batches} batches...", flush=True)

        country_candidates = 0
        country_matches = 0

        for b_idx in range(num_batches):
            b_start_idx = b_idx * s1_batch_size
            b_end_idx = min((b_idx + 1) * s1_batch_size, n_s1_country)
            s1_batch = s1_c.iloc[b_start_idx:b_end_idx].copy().reset_index(drop=True)

            t_batch_start = time.time()

            # Normalization
            s1_batch["norm_name"] = s1_batch["business_name"].apply(normalize_name)
            s1_batch["core_name"] = s1_batch["norm_name"].apply(clean_core_name)
            s1_batch["norm_addr"] = s1_batch["business_address"].apply(normalize_address)
            s1_batch["norm_country"] = s1_batch["country"].apply(normalize_country)

            # Generate candidate keys and retrieve candidates
            s1_keys_list = generate_blocking_keys(s1_batch)
            s1_ids = s1_batch["entity_id"].tolist()

            batch_pairs_s1 = []
            batch_pairs_cand = []
            batch_pairs_src = []
            s1_candidates_dict = {}

            for s1_id, keys in zip(s1_ids, s1_keys_list):
                cand_scores = defaultdict(int)
                cand_source_map = {}

                for k in keys:
                    if k in index.index:
                        cand_list = index.index[k]
                        if len(cand_list) <= index.max_block_size:
                            for cand_id, cand_src in cand_list:
                                cand_scores[cand_id] += 1
                                cand_source_map[cand_id] = cand_src

                sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))
                if max_candidates_per_query and len(sorted_cands) > max_candidates_per_query:
                    sorted_cands = sorted_cands[:max_candidates_per_query]

                c_id_list = []
                for cand_id, _ in sorted_cands:
                    c_id_list.append(cand_id)
                    batch_pairs_s1.append(s1_id)
                    batch_pairs_cand.append(cand_id)
                    batch_pairs_src.append(cand_source_map[cand_id])

                s1_candidates_dict[s1_id] = c_id_list

            num_pairs = len(batch_pairs_s1)
            country_candidates += num_pairs

            if num_pairs > 0:
                cand_df = pd.DataFrame({
                    "source1_id": batch_pairs_s1,
                    "candidate_id": batch_pairs_cand,
                    "candidate_source": batch_pairs_src,
                })

                unique_batch_s1 = set(batch_pairs_s1)
                unique_batch_cands = set(batch_pairs_cand)

                # Load lookups ONLY for records present in this batch
                lookups = load_entity_lookups(
                    s1_df=s1_batch,
                    s2_df=s2_c,
                    s3_df=s3_c,
                    needed_s1_ids=unique_batch_s1,
                    needed_cand_ids=unique_batch_cands,
                )

                # Compute features
                feat_df = generate_feature_matrix(cand_df, lookups)
                del lookups, cand_df
                gc.collect()

                # Inference
                X_batch = feat_df[feature_names].values
                probs = model.predict_proba(X_batch)[:, 1]

                # Aggregate passing matches per S1
                s1_matches_dict = defaultdict(list)
                for s1_id, c_id, prob in zip(batch_pairs_s1, batch_pairs_cand, probs):
                    if prob >= threshold:
                        if c_id.startswith(("S2-", "S3-")):
                            s1_matches_dict[s1_id].append((c_id, float(prob)))

                del feat_df, X_batch
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
                        country_matches += len(m_list)
                    else:
                        matching_map[s1_id] = []
            else:
                for s1_id in s1_ids:
                    candidates_map[s1_id] = []
                    matching_map[s1_id] = []

            b_time = time.time() - t_batch_start
            print(f"    Batch {b_idx+1}/{num_batches} ({len(s1_batch):,} S1s, {num_pairs:,} pairs) done in {b_time:.2f}s", flush=True)

            del s1_batch, s1_keys_list, batch_pairs_s1, batch_pairs_cand, batch_pairs_src, s1_candidates_dict
            gc.collect()

        total_candidates_evaluated += country_candidates
        total_matches_found += country_matches
        print(f"  Finished {country} in {time.time()-c_start:.2f}s | Candidates: {country_candidates:,} | Matches: {country_matches:,}", flush=True)

        del index, s2_c, s3_c, s1_c
        gc.collect()

    del s1_full_df
    gc.collect()

    # 4. Format and write submission files
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
    print("\n[Step 5/6] Prediction statistics...", flush=True)
    non_empty_count = sum(1 for m in matching_map.values() if len(m) > 0)
    empty_count = len(all_required_s1_ids) - non_empty_count

    print(f"  Total S1 Entities:     {len(all_required_s1_ids):,}", flush=True)
    print(f"  Non-empty Predictions: {non_empty_count:,} ({non_empty_count/len(all_required_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Empty Predictions:     {empty_count:,} ({empty_count/len(all_required_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Total Match Links:     {n_links:,}", flush=True)
    print(f"  Threshold Used:        {threshold:.2f}", flush=True)

    # 6. Run official validation check
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
    print(f"TEST INFERENCE AND SUBMISSION CREATION COMPLETED IN {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print("=" * 85, flush=True)

    return {
        "test_s1_count": len(all_required_s1_ids),
        "non_empty": non_empty_count,
        "empty": empty_count,
        "total_links": n_links,
        "threshold": threshold,
        "validator_result": val_status,
        "runtime_sec": total_time,
    }


if __name__ == "__main__":
    run_full_test_inference()
