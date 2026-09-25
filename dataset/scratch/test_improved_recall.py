import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import time
import pandas as pd

from person1.data_loader import load_source, load_ground_truth
from person1.candidate_generation import generate_candidates
from person1.evaluate_blocking import evaluate_candidate_set, analyze_missed_matches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading benchmark dataset...")
s1 = load_source("benchmark/eval_source1.tsv")
s2 = load_source("benchmark/eval_source2.tsv")
s3 = load_source("benchmark/eval_source3.tsv")
gt = load_ground_truth("benchmark/eval_ground_truth.tsv")

print(f"Loaded S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}, GT: {len(gt):,}")

strategies = {"name_prefix_4", "name_prefix_6", "name_tokens", "name_sorted", "name_compressed", "address_key"}

t0 = time.time()
cands_df = generate_candidates(
    source1=s1,
    source2=s2,
    source3=s3,
    strategies=strategies,
    max_block_size=500,
    max_candidates_per_query=200,
)
t_elapsed = time.time() - t0

metrics = evaluate_candidate_set(
    candidate_pairs_df=cands_df,
    ground_truth_df=gt,
    s1_count=len(s1),
    strategy_name="Improved Full Union",
)
metrics["runtime_sec"] = t_elapsed

print("\n" + "=" * 90)
print(f"Improved Union -> Recall: {metrics['recall']*100:.2f}% | Candidates: {metrics['candidate_count']:,} | Avg/S1: {metrics['avg_candidates_per_s1']:.1f} | Time: {t_elapsed:.2f}s")
print("=" * 90)

analyze_missed_matches(
    candidate_pairs_df=cands_df,
    ground_truth_df=gt,
    source1_df=s1,
    source2_df=s2,
    source3_df=s3,
    max_examples=5,
)
