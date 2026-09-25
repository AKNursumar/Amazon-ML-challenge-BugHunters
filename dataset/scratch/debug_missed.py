import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import pandas as pd

from person1.blocking import BlockingIndex, generate_blocking_keys
from person1.candidate_generation import generate_candidates
from person1.normalization import normalize_name, clean_core_name, normalize_address, normalize_country

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

s1 = pd.read_csv("benchmark/eval_source1.tsv", sep="\t")
s2 = pd.read_csv("benchmark/eval_source2.tsv", sep="\t")
s3 = pd.read_csv("benchmark/eval_source3.tsv", sep="\t")

# Check S1-805248801 and S3-824997676 (Rapid Golden Oxley)
s1_row = s1[s1['entity_id'] == 'S1-805248801'].iloc[0]
s3_row = s3[s3['entity_id'] == 'S3-824997676'].iloc[0]

print("S1-805248801:")
print("  Name:", s1_row['business_name'])
print("  Country:", s1_row['country'])

print("S3-824997676:")
print("  Name:", s3_row['business_name'])
print("  Country:", s3_row['country'])

from person1.blocking import get_record_blocking_keys
s1_keys = get_record_blocking_keys(
    normalize_country(s1_row['country']),
    normalize_name(s1_row['business_name']),
    clean_core_name(normalize_name(s1_row['business_name'])),
    normalize_address(s1_row['business_address'])
)
s3_keys = get_record_blocking_keys(
    normalize_country(s3_row['country']),
    normalize_name(s3_row['business_name']),
    clean_core_name(normalize_name(s3_row['business_name'])),
    normalize_address(s3_row['business_address'])
)

print("S1 Keys:", s1_keys)
print("S3 Keys:", s3_keys)
print("Intersection:", s1_keys.intersection(s3_keys))

# Build index on S3
index = BlockingIndex(max_block_size=500)
index.add_candidates(s3, source_name="source3")

for k in s1_keys.intersection(s3_keys):
    print(f"Key '{k}': block size = {len(index.index.get(k, []))}")
