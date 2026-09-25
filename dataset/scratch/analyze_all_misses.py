import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import re
import unicodedata
from collections import Counter, defaultdict
import pandas as pd

from person1.data_loader import load_source, load_ground_truth, get_true_pairs
from person1.normalization import (
    normalize_name, clean_core_name, normalize_address, normalize_country, extract_address_key
)
from person1.blocking import BlockingIndex, generate_blocking_keys, get_record_blocking_keys
from person1.candidate_generation import generate_candidates

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Step 1: Loading benchmark datasets...")
s1 = load_source("benchmark/eval_source1.tsv")
s2 = load_source("benchmark/eval_source2.tsv")
s3 = load_source("benchmark/eval_source3.tsv")
gt = load_ground_truth("benchmark/eval_ground_truth.tsv")

true_pairs = get_true_pairs(gt)
print(f"Loaded S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}, Total True Pairs: {len(true_pairs):,}")

# Precompute normalizations in-place
print("Normalizing records...")
for df in [s1, s2, s3]:
    if "norm_name" not in df.columns:
        df["norm_name"] = df["business_name"].apply(normalize_name)
    if "core_name" not in df.columns:
        df["core_name"] = df["norm_name"].apply(clean_core_name)
    if "norm_addr" not in df.columns:
        df["norm_addr"] = df["business_address"].apply(normalize_address)
    if "norm_country" not in df.columns:
        df["norm_country"] = df["country"].apply(normalize_country)

print("Step 2: Running baseline candidate generation...")
cands_df = generate_candidates(
    source1=s1,
    source2=s2,
    source3=s3,
    max_block_size=500,
    max_candidates_per_query=200
)

cand_pairs = set(zip(cands_df["source1_id"], cands_df["candidate_id"]))
hits = true_pairs.intersection(cand_pairs)
missed_pairs = true_pairs - cand_pairs

baseline_recall = len(hits) / len(true_pairs)
print(f"Baseline Recall: {baseline_recall*100:.2f}% ({len(hits)} / {len(true_pairs)} found)")
print(f"Total Missed True Matches: {len(missed_pairs)} ({len(missed_pairs)/len(true_pairs)*100:.2f}%)")

print("Step 3: Building detailed analysis on missed pairs...")
# Quick lookup maps for sources
needed_s1_ids = {p[0] for p in missed_pairs}
needed_cand_ids = {p[1] for p in missed_pairs}

s1_sub = s1[s1["entity_id"].isin(needed_s1_ids)].set_index("entity_id").to_dict("index")
s2_sub = s2[s2["entity_id"].isin(needed_cand_ids)].set_index("entity_id").to_dict("index")
s3_sub = s3[s3["entity_id"].isin(needed_cand_ids)].set_index("entity_id").to_dict("index")

# We also build an inverted index on S2+S3 to check block sizes of missed keys
print("Building index block size lookup...")
index = BlockingIndex(max_block_size=500)
index.add_candidates(s2, source_name="source2")
index.add_candidates(s3, source_name="source3")

raw_block_sizes = {k: len(v) for k, v in index.index.items()}
index.prune_large_blocks()
pruned_keys_set = set(raw_block_sizes.keys()) - set(index.index.keys())

def is_indic(text):
    for ch in text:
        name = unicodedata.name(ch, '')
        if any(script in name for script in ['DEVANAGARI', 'TAMIL', 'TELUGU', 'BENGALI', 'GUJARATI', 'KANNADA', 'MALAYALAM', 'GURMUKHI']):
            return True
    return False

missed_records = []
category_counter = Counter()

# Strategy pass/fail counters
strategy_pass_counts = Counter()

for s1_id, cand_id in missed_pairs:
    s1_row = s1_sub.get(s1_id, {})
    cand_row = s2_sub.get(cand_id) or s3_sub.get(cand_id) or {}
    cand_src = "source2" if cand_id.startswith("S2-") else "source3"

    s1_orig_name = s1_row.get("business_name", "")
    cand_orig_name = cand_row.get("business_name", "")
    s1_norm_name = s1_row.get("norm_name", "")
    cand_norm_name = cand_row.get("norm_name", "")
    s1_core = s1_row.get("core_name", "")
    cand_core = cand_row.get("core_name", "")

    s1_orig_addr = s1_row.get("business_address", "")
    cand_orig_addr = cand_row.get("business_address", "")
    s1_norm_addr = s1_row.get("norm_addr", "")
    cand_norm_addr = cand_row.get("norm_addr", "")
    country = s1_row.get("country", "")

    # Keys generated for each individual strategy
    k_np4_s1 = get_record_blocking_keys(country, s1_norm_name, s1_core, s1_norm_addr, {"name_prefix_4"})
    k_np4_c = get_record_blocking_keys(country, cand_norm_name, cand_core, cand_norm_addr, {"name_prefix_4"})
    pass_np4 = bool(k_np4_s1.intersection(k_np4_c))

    k_np6_s1 = get_record_blocking_keys(country, s1_norm_name, s1_core, s1_norm_addr, {"name_prefix_6"})
    k_np6_c = get_record_blocking_keys(country, cand_norm_name, cand_core, cand_norm_addr, {"name_prefix_6"})
    pass_np6 = bool(k_np6_s1.intersection(k_np6_c))

    k_tok_s1 = get_record_blocking_keys(country, s1_norm_name, s1_core, s1_norm_addr, {"name_tokens"})
    k_tok_c = get_record_blocking_keys(country, cand_norm_name, cand_core, cand_norm_addr, {"name_tokens"})
    pass_tok = bool(k_tok_s1.intersection(k_tok_c))

    k_sort_s1 = get_record_blocking_keys(country, s1_norm_name, s1_core, s1_norm_addr, {"name_sorted"})
    k_sort_c = get_record_blocking_keys(country, cand_norm_name, cand_core, cand_norm_addr, {"name_sorted"})
    pass_sort = bool(k_sort_s1.intersection(k_sort_c))

    k_comp_s1 = get_record_blocking_keys(country, s1_norm_name, s1_core, s1_norm_addr, {"name_compressed"})
    k_comp_c = get_record_blocking_keys(country, cand_norm_name, cand_core, cand_norm_addr, {"name_compressed"})
    pass_comp = bool(k_comp_s1.intersection(k_comp_c))

    k_addr_s1 = get_record_blocking_keys(country, s1_norm_name, s1_core, s1_norm_addr, {"address_key"})
    k_addr_c = get_record_blocking_keys(country, cand_norm_name, cand_core, cand_norm_addr, {"address_key"})
    pass_addr = bool(k_addr_s1.intersection(k_addr_c))

    all_k_s1 = k_np4_s1 | k_np6_s1 | k_tok_s1 | k_sort_s1 | k_comp_s1 | k_addr_s1
    all_k_c = k_np4_c | k_np6_c | k_tok_c | k_sort_c | k_comp_c | k_addr_c
    common_keys = all_k_s1.intersection(all_k_c)

    if pass_np4: strategy_pass_counts["name_prefix_4"] += 1
    if pass_np6: strategy_pass_counts["name_prefix_6"] += 1
    if pass_tok: strategy_pass_counts["name_tokens"] += 1
    if pass_sort: strategy_pass_counts["name_sorted"] += 1
    if pass_comp: strategy_pass_counts["name_compressed"] += 1
    if pass_addr: strategy_pass_counts["address_key"] += 1

    # Diagnosis: Why did this pair not make it into the candidate set?
    failure_reason = ""
    category = ""

    if len(common_keys) > 0:
        # A common key existed! Why wasn't it generated?
        # Was it pruned due to block size > 500?
        pruned_common = [k for k in common_keys if k in pruned_keys_set]
        survived_common = [k for k in common_keys if k not in pruned_keys_set]
        if pruned_common and not survived_common:
            category = "Key pruned (Block size > 500)"
            failure_reason = f"All shared keys ({pruned_common}) exceeded max_block_size"
        else:
            category = "Truncated by max_candidates_per_query"
            failure_reason = f"Shared key survived ({survived_common}), but candidate rank exceeded top-200"
    else:
        # No common key existed at all! Determine failure category from text properties
        if is_indic(cand_orig_name) and not is_indic(s1_orig_name):
            category = "Indic script vs Latin name"
            failure_reason = "Cand name is in Indic script; no shared address key generated"
        elif not cand_orig_addr or cand_orig_addr.lower() in ("nan", "none", "null", ""):
            category = "Candidate address missing"
            failure_reason = "Cand address is missing (nan), and names share no blocking key"
        elif not cand_orig_name or cand_orig_name.lower() in ("nan", "none", "null", ""):
            category = "Candidate name missing"
            failure_reason = "Cand name is missing, and address keys did not match"
        elif len(s1_core.replace(" ", "")) <= 4 or len(cand_core.replace(" ", "")) <= 4:
            category = "Short business name"
            failure_reason = "Short name (<4 chars) produced no prefix/token matches"
        elif re.sub(r'[^a-z0-9]', '', s1_core) == re.sub(r'[^a-z0-9]', '', cand_core):
            category = "Exact name without punctuation/spaces"
            failure_reason = "Names match when alphanumeric only, but prefix/token failed"
        else:
            # Check address similarity vs name similarity
            s1_addr_tokens = set(s1_norm_addr.split())
            c_addr_tokens = set(cand_norm_addr.split())
            shared_addr = s1_addr_tokens.intersection(c_addr_tokens) - {"rd", "st", "ave", "dr", "ln", "us", "in", "tn", "nc", "fl", "oh"}
            
            s1_name_tokens = set(s1_core.split())
            c_name_tokens = set(cand_core.split())
            shared_name = s1_name_tokens.intersection(c_name_tokens)

            if len(shared_addr) >= 2 and len(shared_name) == 0:
                category = "Address similar, Name alias/mismatch"
                failure_reason = f"Address shares tokens {shared_addr}, but names share 0 tokens"
            elif len(s1_name_tokens) > 0 and len(c_name_tokens) > 0:
                # Check character 3-gram overlap or spelling typo
                s1_clean = re.sub(r'[^a-z]', '', s1_core)
                c_clean = re.sub(r'[^a-z]', '', cand_core)
                if s1_clean[:3] == c_clean[:3]:
                    category = "Prefix-3 match (Typo after 3 chars)"
                    failure_reason = f"Names share first 3 chars ('{s1_clean[:3]}') but diverge at char 4"
                elif any(tok in cand_orig_name.lower() for tok in s1_name_tokens if len(tok) >= 3):
                    category = "Token variation / Substring"
                    failure_reason = "Sub-token present in raw candidate name but missed in core name"
                else:
                    category = "Severe name & address variation"
                    failure_reason = "Both name and address have extensive structural differences"
            else:
                category = "Other variation"
                failure_reason = "Unclassified variation"

    category_counter[category] += 1

    missed_records.append({
        "source1_id": s1_id,
        "true_candidate_id": cand_id,
        "candidate_source": cand_src,
        "country": country,
        "source1_name": s1_orig_name,
        "true_candidate_name": cand_orig_name,
        "source1_address": s1_orig_addr,
        "true_candidate_address": cand_orig_addr,
        "source1_core_name": s1_core,
        "candidate_core_name": cand_core,
        "pass_name_prefix_4": pass_np4,
        "pass_name_prefix_6": pass_np6,
        "pass_name_tokens": pass_tok,
        "pass_name_sorted": pass_sort,
        "pass_name_compressed": pass_comp,
        "pass_address_key": pass_addr,
        "common_keys": ";".join(common_keys),
        "failure_category": category,
        "failure_reason": failure_reason,
    })

# Save to CSV
analysis_df = pd.DataFrame(missed_records)
out_csv = "blocking_error_analysis.csv"
analysis_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"Saved complete missed matches analysis to: {out_csv} ({len(analysis_df):,} rows)")

print("\n" + "=" * 80)
print("CLASSIFICATION OF MISSED TRUE MATCHES (1,974 TOTAL):")
print("=" * 80)
for cat, count in category_counter.most_common():
    pct = count / len(missed_records) * 100
    print(f"  {cat:<45}: {count:>5} ({pct:>5.1f}%)")

print("\n" + "=" * 80)
print("KEYS MATCHED AMONG MISSED PAIRS (Keys generated but blocked/truncated):")
print("=" * 80)
for strat, pcount in strategy_pass_counts.most_common():
    print(f"  Passed {strat:<25}: {pcount:>5} ({pcount/len(missed_records)*100:>5.1f}%)")
