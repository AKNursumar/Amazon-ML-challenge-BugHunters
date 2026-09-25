import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("blocking_error_analysis.csv")

categories = [
    "Key pruned (Block size > 500)",
    "Indic script vs Latin name",
    "Truncated by max_candidates_per_query",
    "Address similar, Name alias/mismatch",
    "Prefix-3 match (Typo after 3 chars)"
]

for cat in categories:
    print("=" * 80)
    print("CATEGORY:", cat)
    print("=" * 80)
    sub = df[df['failure_category'] == cat].head(3)
    for _, r in sub.iterrows():
        print(f"S1 [{r['source1_id']}]: {r['source1_name']}")
        print(f"  S1 Addr:   {r['source1_address']}")
        print(f"Cand [{r['true_candidate_id']}]: {r['true_candidate_name']}")
        print(f"  Cand Addr: {r['true_candidate_address']}")
        print(f"  Reason:    {r['failure_reason']}")
        print("-" * 60)
