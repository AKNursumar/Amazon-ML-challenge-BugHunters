import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import time
from collections import defaultdict
import pandas as pd

from person1.data_loader import load_source, load_ground_truth, get_true_pairs
from person1.normalization import (
    normalize_name, clean_core_name, normalize_address, normalize_country, extract_address_key
)
from person1.blocking import BlockingIndex, generate_blocking_keys, get_record_blocking_keys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading benchmark dataset...")
s1 = load_source("benchmark/eval_source1.tsv")
s2 = load_source("benchmark/eval_source2.tsv")
s3 = load_source("benchmark/eval_source3.tsv")
gt = load_ground_truth("benchmark/eval_ground_truth.tsv")

true_pairs = get_true_pairs(gt)
total_true = len(true_pairs)
print(f"Total True Pairs: {total_true:,}")

# Precompute normalized columns in-place
print("Precomputing normalizations...")
for df in [s1, s2, s3]:
    if "norm_name" not in df.columns:
        df["norm_name"] = df["business_name"].apply(normalize_name)
    if "core_name" not in df.columns:
        df["core_name"] = df["norm_name"].apply(clean_core_name)
    if "norm_addr" not in df.columns:
        df["norm_addr"] = df["business_address"].apply(normalize_address)
    if "norm_country" not in df.columns:
        df["norm_country"] = df["country"].apply(normalize_country)

print("Building master index on Source 2 and Source 3 (Done ONCE)...")
t0 = time.time()

# Master inverted index without size pruning
master_index = defaultdict(list)

# We can also add new candidate strategy keys to test their individual impact!
def get_extended_keys(country, norm_name, core_name, norm_addr):
    keys = set()
    if not country: country = "UNKNOWN"
    compressed_core = core_name.replace(" ", "")

    # Base strategies
    if len(compressed_core) >= 4:
        keys.add(f"{country}|NP4|{compressed_core[:4]}")
    if len(compressed_core) >= 6:
        keys.add(f"{country}|NP6|{compressed_core[:6]}")
    if len(compressed_core) >= 5:
        keys.add(f"{country}|NCOMP|{compressed_core}")

    tokens = core_name.split()
    for tok in tokens:
        if len(tok) >= 4:
            keys.add(f"{country}|TOK|{tok}")
        # NEW STRATEGY 1: 3-letter acronyms / distinctive tokens (e.g. ONP, PUN, TXO)
        elif len(tok) == 3 and tok.isalpha() and tok not in {"and", "the", "for", "ltd", "inc", "llc", "pvt", "corp", "co"}:
            keys.add(f"{country}|TOK3|{tok}")

    if len(tokens) >= 2:
        sig_tokens = [t for t in tokens if len(t) >= 3 and t not in {"and", "the", "for", "ltd", "inc", "llc", "pvt"}]
        if len(sig_tokens) >= 2:
            sorted_key = "_".join(sorted(sig_tokens[:4]))
            keys.add(f"{country}|NSORT|{sorted_key}")

    if norm_addr:
        addr_k = extract_address_key(norm_addr)
        if addr_k and len(addr_k) >= 3:
            keys.add(f"{country}|ADDR|{addr_k}")

        addr_tokens = norm_addr.split()
        for tok in addr_tokens:
            clean_tok = tok.replace("-", "").replace("/", "")
            if len(clean_tok) >= 3 and clean_tok[-2:] in {"st", "nd", "rd", "th"} and clean_tok[:-2].isdigit():
                continue
            if any(c.isdigit() for c in clean_tok) and any(c.isalpha() for c in clean_tok) and 3 <= len(clean_tok) <= 8:
                keys.add(f"{country}|UNIT|{clean_tok}")
            elif "/" in tok and any(c.isdigit() for c in tok) and len(tok) <= 8:
                keys.add(f"{country}|PLOT|{tok}")

        # NEW STRATEGY 2: Non-numeric address key (for Indic / highway addresses without house numbers)
        # Find 2 significant locality tokens
        ignored_addr = {"rd", "st", "ave", "dr", "ln", "hwy", "blvd", "ct", "cir", "fl", "apt", "ste", "unit", "shop", "near", "opp", "opposite", "behind", "hotel", "road", "street", "highway"}
        sig_addr_words = [w for w in addr_tokens if len(w) >= 4 and w not in ignored_addr and w.isalpha()]
        if len(sig_addr_words) >= 2:
            keys.add(f"{country}|ADDR_LOC|{sig_addr_words[0]}_{sig_addr_words[1]}")

    return keys

# Index S2
for eid, c, nn, cn, na in zip(s2["entity_id"], s2["norm_country"], s2["norm_name"], s2["core_name"], s2["norm_addr"]):
    item = (eid, "source2")
    for k in get_extended_keys(c, nn, cn, na):
        master_index[k].append(item)

# Index S3
for eid, c, nn, cn, na in zip(s3["entity_id"], s3["norm_country"], s3["norm_name"], s3["core_name"], s3["norm_addr"]):
    item = (eid, "source3")
    for k in get_extended_keys(c, nn, cn, na):
        master_index[k].append(item)

print(f"Master index built in {time.time() - t0:.1f}s. Total unique keys: {len(master_index):,}")

# Precompute S1 keys
s1_keys_all = []
for c, nn, cn, na in zip(s1["norm_country"], s1["norm_name"], s1["core_name"], s1["norm_addr"]):
    s1_keys_all.append(get_extended_keys(c, nn, cn, na))

s1_ids = s1["entity_id"].tolist()

def query_candidates(allowed_prefixes, max_block_size, max_cands_per_s1):
    records = []
    for s1_id, keys in zip(s1_ids, s1_keys_all):
        cand_scores = defaultdict(int)
        cand_source_map = {}

        for k in keys:
            # Check strategy prefix filter
            prefix = k.split("|")[1] if "|" in k else ""
            if allowed_prefixes is not None and prefix not in allowed_prefixes:
                continue

            cand_list = master_index.get(k, [])
            if 0 < len(cand_list) <= max_block_size:
                for cand_id, cand_src in cand_list:
                    cand_scores[cand_id] += 1
                    cand_source_map[cand_id] = cand_src

        sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))
        if max_cands_per_s1 and len(sorted_cands) > max_cands_per_s1:
            sorted_cands = sorted_cands[:max_cands_per_s1]

        for cand_id, _ in sorted_cands:
            records.append((s1_id, cand_id))

    return set(records)

experiments = [
    # 1. Baseline configuration
    ("Baseline (Current)", {"NP4", "NP6", "TOK", "NSORT", "NCOMP", "ADDR", "UNIT", "PLOT"}, 500, 200),
    # 2. Standalone New Strategy: 3-letter Acronyms only
    ("Standalone: TOK3 (3-Letter Acronyms)", {"TOK3"}, 1000, 200),
    # 3. Standalone New Strategy: ADDR_LOC (Non-numeric Locality) only
    ("Standalone: ADDR_LOC (Locality Pair)", {"ADDR_LOC"}, 1000, 200),
    # 4. Standalone: Base Address Keys only
    ("Standalone: Base Address (ADDR+UNIT+PLOT)", {"ADDR", "UNIT", "PLOT"}, 1000, 200),
    # 5. Standalone: Base Name Keys only
    ("Standalone: Base Name (NP4+NP6+TOK+NSORT+NCOMP)", {"NP4", "NP6", "TOK", "NSORT", "NCOMP"}, 1000, 200),
    # 6. Parameter Tuning: Increase max_block_size to 1000
    ("Tuning A: Baseline + max_block_size=1000", {"NP4", "NP6", "TOK", "NSORT", "NCOMP", "ADDR", "UNIT", "PLOT"}, 1000, 200),
    # 7. Parameter Tuning: Increase max_cands_per_s1 to 300
    ("Tuning B: Baseline + max_block_size=1000 + max_cands=300", {"NP4", "NP6", "TOK", "NSORT", "NCOMP", "ADDR", "UNIT", "PLOT"}, 1000, 300),
    # 8. Combined V1: Base + TOK3 (Acronyms)
    ("Combined V1: Base + TOK3", {"NP4", "NP6", "TOK", "NSORT", "NCOMP", "ADDR", "UNIT", "PLOT", "TOK3"}, 1000, 300),
    # 9. Combined V2: Base + TOK3 + ADDR_LOC (Locality)
    ("Combined V2: Base + TOK3 + ADDR_LOC", {"NP4", "NP6", "TOK", "NSORT", "NCOMP", "ADDR", "UNIT", "PLOT", "TOK3", "ADDR_LOC"}, 1000, 300),
    # 10. Combined V3 (Optimal High-Recall): Base + TOK3 + ADDR_LOC with max_block=1200, max_cands=350
    ("Combined V3 (High Recall): Base + All New + Tuning", {"NP4", "NP6", "TOK", "NSORT", "NCOMP", "ADDR", "UNIT", "PLOT", "TOK3", "ADDR_LOC"}, 1200, 350),
]

baseline_set = None
results = []

for name, allowed_pfx, max_b, max_c in experiments:
    t1 = time.time()
    c_set = query_candidates(allowed_pfx, max_b, max_c)
    t_el = time.time() - t1

    hits = len(c_set.intersection(true_pairs))
    rec = hits / total_true
    cnt = len(c_set)
    avg_s1 = cnt / len(s1)
    reduction = (1 - (cnt / (len(s1) * (len(s2) + len(s3))))) * 100

    if baseline_set is None:
        baseline_set = c_set
        recovered_over_base = 0
    else:
        # How many true matches in c_set were NOT in baseline_set?
        recovered_over_base = len((c_set - baseline_set).intersection(true_pairs))

    results.append({
        "Experiment": name,
        "Recall": rec,
        "Hits": hits,
        "Recovered": recovered_over_base,
        "Candidates": cnt,
        "Avg/S1": avg_s1,
        "Reduction": reduction,
        "QueryTime": t_el
    })
    print(f"[{name}]")
    print(f"  Recall: {rec*100:.2f}% ({hits:,}/{total_true:,}) | New Recovered: +{recovered_over_base} | Candidates: {cnt:,} (Avg/S1: {avg_s1:.1f}) | Time: {t_el:.2f}s\n")

print("=" * 120)
print(f"{'Experiment':<52} | {'Recall':<8} | {'Recovered':<10} | {'Candidates':<12} | {'Avg/S1':<8} | {'Reduction':<10}")
print("-" * 120)
for r in results:
    print(f"{r['Experiment']:<52} | {r['Recall']*100:>6.2f}% | {r['Recovered']:>10} | {r['Candidates']:>12,} | {r['Avg/S1']:>8.1f} | {r['Reduction']:>9.4f}%")
print("=" * 120)
