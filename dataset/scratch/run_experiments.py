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

experiments = [
    ("Exp A: Name Prefix 4", {"name_prefix_4"}),
    ("Exp B: Name Prefix 6", {"name_prefix_6"}),
    ("Exp C: Name Tokens", {"name_tokens"}),
    ("Exp D: Name Sorted Tokens", {"name_sorted"}),
    ("Exp E: Name Compressed/Domain", {"name_compressed"}),
    ("Exp F: Address Tokens", {"address_key"}),
    ("Exp G: Name Prefix + Tokens", {"name_prefix_4", "name_tokens"}),
    ("Exp H: Combined (Prefix + Tokens + Address)", {"name_prefix_4", "name_tokens", "address_key"}),
    ("Exp I: Full Union (Prefix4+6 + Tokens + Sorted + Comp + Addr)", {
        "name_prefix_4", "name_prefix_6", "name_tokens", "name_sorted", "name_compressed", "address_key"
    }),
]

results = []

last_candidates_df = None
last_strategy_name = None

for exp_name, strategies in experiments:
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
        strategy_name=exp_name,
    )
    metrics["runtime_sec"] = t_elapsed
    results.append(metrics)
    
    last_candidates_df = cands_df
    last_strategy_name = exp_name
    
    print(f"[{exp_name}] -> Recall: {metrics['recall']*100:.2f}% | Candidates: {metrics['candidate_count']:,} | Avg/S1: {metrics['avg_candidates_per_s1']:.1f} | Time: {t_elapsed:.2f}s")

# Print formatted experiment table
print("\n" + "=" * 90)
print(f"{'Strategy':<45} | {'Recall':<8} | {'Candidates':<12} | {'Avg/S1':<8} | {'Time (s)':<8}")
print("-" * 90)
for r in results:
    print(f"{r['strategy']:<45} | {r['recall']*100:>6.2f}% | {r['candidate_count']:>12,} | {r['avg_candidates_per_s1']:>8.1f} | {r['runtime_sec']:>8.2f}")
print("=" * 90)

# Error analysis on the best combined strategy
analyze_missed_matches(
    candidate_pairs_df=last_candidates_df,
    ground_truth_df=gt,
    source1_df=s1,
    source2_df=s2,
    source3_df=s3,
    max_examples=10,
)
