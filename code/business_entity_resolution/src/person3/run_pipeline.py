"""
Pipeline CLI and Automation Runner for Person 3.
Amazon ML Challenge 2026.

Supports:
1. Validation Mode (--mode val):
   - Evaluates 90/10 and 80/20 group splits on source1_id.
   - Computes macro F0.5, macro precision, macro recall, and singleton accuracy.
   - Generates threshold tuning curves and diagnostic error reports.
2. Full / Test Submission Mode (--mode test):
   - Generates final output/matching_results.tsv and output/candidate_pairs.tsv.
   - Handles multi-matches (e.g. S1-001 -> S2-010,S2-024,S3-091) and singletons (S1-002 -> "").
   - Runs utils/validate_submission.py to guarantee submission compliance.
"""

import argparse
import os
import sys
import io
import time
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
for p in [cur_dir, dataset_dir, repo_root]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from person3.metrics import evaluate_entity_resolution
from person3.decision import DecisionEngine, format_submission_files
from person3.validation import run_validation_experiments
from person3.validate_submission_runner import run_validator


def generate_submission(
    predictions_path: str,
    candidate_pairs_path: str,
    test_source1_path: str,
    output_matching_path: str = "output/matching_results.tsv",
    output_candidate_path: str = "output/candidate_pairs.tsv",
    threshold: float = 0.75,
    run_validation_check: bool = True,
    test_dir: str = None,
):
    """
    Generate final matching_results.tsv and candidate_pairs.tsv from candidate predictions,
    ensuring all test Source 1 entities have exactly one row, and validating the submission.
    """
    t0 = time.time()
    print("=" * 85)
    print("PERSON 3: GENERATING FINAL COMPETITION SUBMISSION FILES")
    print(f"Optimal Decision Threshold: T = {threshold:.2f}")
    print(f"Matching Results Output:    {output_matching_path}")
    print(f"Candidate Pairs Output:     {output_candidate_path}")
    print("=" * 85)

    # 1. Read required S1 entities from test_source1.tsv
    print(f"\n[1/4] Reading required Source 1 entities from: {test_source1_path}")
    s1_ids = []
    with open(test_source1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line:
                s1_ids.append(line.split("\t", 1)[0].strip())
    print(f"  Total required S1 entities: {len(s1_ids):,}")

    # 2. Load candidate predictions
    print(f"\n[2/4] Loading candidate predictions from: {predictions_path}")
    pred_df = pd.read_csv(predictions_path, sep="\t")
    print(f"  Loaded {len(pred_df):,} candidate predictions across {pred_df['source1_id'].nunique():,} unique S1 IDs.")

    # 3. Load or build candidate pairs map
    engine = DecisionEngine(threshold=threshold)

    print("\n[3/4] Aggregating candidates and applying singleton / multiple-match logic...")
    cand_pairs_df = pd.read_csv(candidate_pairs_path, sep="\t")
    candidates_map = engine.aggregate_candidates(cand_pairs_df, s1_ids)

    # Filter matches above threshold
    matching_map = engine.filter_and_aggregate_matches(
        pred_df, s1_ids, custom_threshold=threshold
    )

    # Metrics on the generated set
    matched_s1 = sum(1 for m in matching_map.values() if len(m) > 0)
    singleton_s1 = len(s1_ids) - matched_s1
    total_matches = sum(len(m) for m in matching_map.values())
    print(f"  Matched S1 entities:  {matched_s1:,} ({matched_s1/len(s1_ids)*100:.2f}%)")
    print(f"  Singleton entities:   {singleton_s1:,} ({singleton_s1/len(s1_ids)*100:.2f}%)")
    print(f"  Total matched links:  {total_matches:,}")

    # 4. Write submission files
    print(f"\n[4/4] Writing tab-separated submission files...")
    n_rows, _ = format_submission_files(
        matching_map=matching_map,
        candidates_map=candidates_map,
        required_s1_ids=s1_ids,
        matching_output_path=output_matching_path,
        candidate_output_path=output_candidate_path,
    )

    match_mb = os.path.getsize(output_matching_path) / (1024 * 1024)
    cand_mb = os.path.getsize(output_candidate_path) / (1024 * 1024)
    print(f"  Created: {output_matching_path} ({match_mb:.2f} MB, {n_rows:,} rows)")
    print(f"  Created: {output_candidate_path} ({cand_mb:.2f} MB, {n_rows:,} rows)")
    print(f"  Generation completed in {time.time()-t0:.2f}s")

    # 5. Run official validator
    if run_validation_check:
        print("\nRunning submission validator check...")
        if test_dir is None:
            test_dir = os.path.dirname(os.path.abspath(test_source1_path))
        run_validator(
            matching_path=output_matching_path,
            candidate_path=output_candidate_path,
            test_dir=test_dir,
            check_ids=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Person 3 Final Pipeline Runner")
    parser.add_argument("--mode", choices=["val", "test", "both"], default="both", help="Execution mode")
    parser.add_argument("--threshold", type=float, default=0.75, help="Optimal decision threshold (default: 0.75)")
    parser.add_argument("--matching-out", default="output/matching_results.tsv", help="Matching results output path")
    parser.add_argument("--candidate-out", default="output/candidate_pairs.tsv", help="Candidate pairs output path")
    args = parser.parse_args()

    # Paths
    preds_path = os.path.join(dataset_dir, "person2", "predictions", "candidate_predictions.tsv")
    cands_path = os.path.join(dataset_dir, "candidate_pairs.tsv")
    train_gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

    test_s1_path = os.path.join(dataset_dir, "test", "test_source1.tsv")
    if not os.path.exists(test_s1_path):
        test_s1_path = os.path.join(repo_root, "Resources", "student_resource", "dataset", "test", "test_source1.tsv")

    if args.mode in ["val", "both"]:
        print("\n>>> EXECUTING VALIDATION BENCHMARK EXPERIMENTS...")
        run_validation_experiments(
            candidate_predictions_path=preds_path,
            ground_truth_path=train_gt_path,
        )

    if args.mode in ["test", "both"]:
        print("\n>>> GENERATING FINAL SUBMISSION FILES...")
        generate_submission(
            predictions_path=preds_path,
            candidate_pairs_path=cands_path,
            test_source1_path=test_s1_path,
            output_matching_path=args.matching_out,
            output_candidate_path=args.candidate_out,
            threshold=args.threshold,
            run_validation_check=True,
        )


if __name__ == "__main__":
    main()
