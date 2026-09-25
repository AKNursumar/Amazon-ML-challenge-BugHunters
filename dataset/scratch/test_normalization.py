import sys
import os
sys.path.insert(0, os.path.abspath("."))
import io
from person1.normalization import (
    normalize_name,
    clean_core_name,
    normalize_address,
    extract_address_key,
    normalize_country
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

test_names = [
    "ABC Pvt. Ltd.",
    "ABC PRIVATE LIMITED",
    "A.B.C PVT LTD",
    "Maure Williams Colombier Inc",
    "maurewilliamscolombier.com",
    "Halodelta aka Clemons Silver Eastern Inc",
    ">> Changodar Sevcis",
    "M/s punia (india) investment private limited",
    "Payne Énterprises",
    "PAYNE-ENRTPRMISES",
    "BS Projects Ltd Ltd",
    "SS Food Private Limited"
]

print("=== TESTING NAME NORMALIZATION ===")
for name in test_names:
    norm = normalize_name(name)
    core = clean_core_name(norm)
    print(f"Original: {name:45} | Norm: {norm:35} | Core: {core}")

test_addrs = [
    "1795 Westchester Drive, High Point, NC",
    "3315 Fremont Street, Peoria, IL",
    "3315 FREMONT SAINT, PEORIA, IL",
    "3315 Fremont St, Peoria, Illinois",
    "1619 Julia Park Drive, Spring, TX",
    "1619 1/2 JULIA PARK DRIVE, SPRING, TX",
    "0189 LAUREL ROAD, ARDEN, NC",
    "189 Laurel Road, Arden, North Carolina",
    "Af-684, Nandgram Near Mother India Public School. Ph. 989, Ghaziabad, Uttar Pradesh",
    "AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. Ghaziabad, उत्तर प्रदेश",
    "Shop No.- 4, Ground Floor, 59/101 Kanhaiya Plaza, Canal Road, Kanpur, Uttar Pradesh",
    "7503 Laytonia Drive, Gaithersburg, MD",
    "7503  LAYTONIA DR, GAITHERSBURG, MD"
]

print("\n=== TESTING ADDRESS NORMALIZATION ===")
for addr in test_addrs:
    norm = normalize_address(addr)
    key = extract_address_key(norm)
    print(f"Original: {addr:65} | Key: {key:20} | Norm: {norm[:40]}...")
