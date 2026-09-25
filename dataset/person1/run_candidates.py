"""
Pipeline Runner Module for Person 1: Candidate Generation.
Amazon ML Challenge 2026.

Can be run directly via CLI to generate candidate_pairs.tsv:
    python person1/run_candidates.py --source1 train/train_source1.tsv --source2 train/train_source2.tsv --source3 train/train_source3.tsv --output candidate_pairs.tsv
"""

import argparse
import os
import sys
import time
import pandas as pd

from .candidate_generation import generate_candidates
from .data_loader import load_source


def main():
    parser = argparse.ArgumentParser(description="Person 1 Candidate Generation Pipeline")
    parser.add_argument("--source1", type=str, default="benchmark/eval_source1.tsv", help="Path to Source 1 TSV")
    parser.add_argument("--source2", type=str, default="benchmark/eval_source2.tsv", help="Path to Source 2 TSV")
    parser.add_argument("--source3", type=str, default="benchmark/eval_source3.tsv", help="Path to Source 3 TSV")
    parser.add_argument("--output", type=str, default="candidate_pairs.tsv", help="Path to output TSV")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows from Source 1 for quick run")
    parser.add_argument("--max_cands", type=int, default=300, help="Max candidates per query")
    parser.add_argument("--max_block_size", type=int, default=1000, help="Max candidates per block before pruning")

    args = parser.parse_args()

    print("=" * 80)
    print("PERSON 1: CANDIDATE GENERATION PIPELINE")
    print(f"Source 1: {args.source1}")
    print(f"Source 2: {args.source2}")
    print(f"Source 3: {args.source3}")
    print(f"Output:   {args.output}")
    print("=" * 80)

    t_start = time.time()

    print("\n[Step 1/4] Loading input datasets...")
    s1 = load_source(args.source1, nrows=args.sample)
    s2 = load_source(args.source2)
    s3 = load_source(args.source3)
    print(f"  Loaded Source 1: {len(s1):,} records")
    print(f"  Loaded Source 2: {len(s2):,} records")
    print(f"  Loaded Source 3: {len(s3):,} records")

    print("\n[Step 2/4] Executing Normalization, Blocking, and Candidate Generation...")
    t_gen_start = time.time()
    candidate_pairs = generate_candidates(
        source1=s1,
        source2=s2,
        source3=s3,
        max_block_size=args.max_block_size,
        max_candidates_per_query=args.max_cands,
    )
    t_gen_elapsed = time.time() - t_gen_start
    print(f"  Generated {len(candidate_pairs):,} candidate pairs in {t_gen_elapsed:.2f}s")
    print(f"  Average candidates per Source 1 entity: {len(candidate_pairs) / max(1, len(s1)):.1f}")

    print("\n[Step 3/4] Validating Candidate Pair Schema and Integrity...")
    expected_cols = ["source1_id", "candidate_id", "candidate_source"]
    assert list(candidate_pairs.columns) == expected_cols, f"Schema mismatch: {candidate_pairs.columns}"
    assert candidate_pairs["source1_id"].isnull().sum() == 0, "Found nulls in source1_id"
    assert candidate_pairs["candidate_id"].isnull().sum() == 0, "Found nulls in candidate_id"
    assert candidate_pairs["candidate_source"].isnull().sum() == 0, "Found nulls in candidate_source"
    print("  Schema validated successfully:", expected_cols)

    print("\n[Step 4/4] Writing candidate pairs to TSV...")
    candidate_pairs.to_csv(args.output, sep="\t", index=False)
    print(f"  Candidate pairs saved to: {args.output} ({os.path.getsize(args.output) / (1024*1024):.2f} MB)")

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
