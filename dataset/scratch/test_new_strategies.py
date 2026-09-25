import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import time
import pandas as pd

from person1.data_loader import load_source, load_ground_truth, get_true_pairs
from person1.normalization import (
    normalize_name, clean_core_name, normalize_address, normalize_country
)
from person1.candidate_generation import generate_candidates
from person1.evaluate_blocking import evaluate_candidate_set

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading benchmark dataset...")
s1 = load_source("benchmark/eval_source1.tsv")
s2 = load_source("benchmark/eval_source2.tsv")
s3 = load_source("benchmark/eval_source3.tsv")
gt = load_ground_truth("benchmark/eval_ground_truth.tsv")

true_pairs = get_true_pairs(gt)
total_true = len(true_pairs)
print(f"Total True Pairs: {total_true:,}")

# Precompute normalized columns
for df in [s1, s2, s3]:
    if "norm_name" not in df.columns:
        df["norm_name"] = df["business_name"].apply(normalize_name)
    if "core_name" not in df.columns:
        df["core_name"] = df["norm_name"].apply(clean_core_name)
    if "norm_addr" not in df.columns:
        df["norm_addr"] = df["business_address"].apply(normalize_address)
    if "norm_country" not in df.columns:
        df["norm_country"] = df["country"].apply(normalize_country)

baseline_recall = 0.9008
baseline_cands = 583_398

experiments = [
    ("Baseline (Current)", None, 500, 200),
    ("Var 1: Increase max_block_size=1000 & max_cands=300", None, 1000, 300),
    ("Var 2: Increase max_block_size=1200 & max_cands=350", None, 1200, 350),
    ("Var 3: Only address_key + plot_key (Standalone)", {"address_key"}, 1000, 300),
    ("Var 4: Only name_tokens (Standalone)", {"name_tokens"}, 1000, 300),
    ("Var 5: Only name_prefix_6 (Standalone)", {"name_prefix_6"}, 1000, 300),
]

results = []

for name, strats, max_block, max_cands in experiments:
    t0 = time.time()
    c_df = generate_candidates(
        source1=s1,
        source2=s2,
        source3=s3,
        strategies=strats,
        max_block_size=max_block,
        max_candidates_per_query=max_cands,
    )
    t_el = time.time() - t0
    
    cand_pairs = set(zip(c_df["source1_id"], c_df["candidate_id"]))
    hits = len(cand_pairs.intersection(true_pairs))
    rec = hits / total_true
    cnt = len(c_df)
    avg_s1 = cnt / len(s1)
    reduction = (1 - (cnt / (len(s1) * (len(s2) + len(s3))))) * 100

    results.append({
        "Experiment": name,
        "Recall": rec,
        "Hits": hits,
        "Candidates": cnt,
        "Avg/S1": avg_s1,
        "Reduction": reduction,
        "Time": t_el,
    })
    print(f"[{name}] -> Recall: {rec*100:.2f}% ({hits:,}/{total_true:,}) | Candidates: {cnt:,} | Avg/S1: {avg_s1:.1f} | Reduction: {reduction:.4f}% | Time: {t_el:.1f}s")

print("\n" + "=" * 105)
print(f"{'Experiment':<50} | {'Recall':<8} | {'Candidates':<12} | {'Avg/S1':<8} | {'Reduction':<10} | {'Time (s)':<8}")
print("-" * 105)
for r in results:
    print(f"{r['Experiment']:<50} | {r['Recall']*100:>6.2f}% | {r['Candidates']:>12,} | {r['Avg/S1']:>8.1f} | {r['Reduction']:>9.4f}% | {r['Time']:>8.1f}")
print("=" * 105)
