"""
Submission Validator Runner and Auditor (Person 3).
Amazon ML Challenge 2026.

Wraps the challenge's official utils/validate_submission.py tool.
Validates:
1. output/matching_results.tsv
2. output/candidate_pairs.tsv
against:
dataset/test/test_source1.tsv (and optionally test_source2.tsv, test_source3.tsv)

Ensures that every formatting, cardinality, and ID constraint is met before submission.
"""

import os
import sys
import io
import subprocess
import argparse
from typing import Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
workspace_root = os.path.dirname(repo_root)


def run_validator(
    matching_path: str = "output/matching_results.tsv",
    candidate_path: Optional[str] = "output/candidate_pairs.tsv",
    test_dir: Optional[str] = None,
    check_ids: bool = False,
    validator_script: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Run utils/validate_submission.py and capture the exact stdout/stderr and exit code.

    Args:
        matching_path: Path to matching_results.tsv.
        candidate_path: Path to candidate_pairs.tsv.
        test_dir: Path to directory containing test_source1.tsv.
        check_ids: If True, passes --check-ids to check S2/S3 ID validity.
        validator_script: Path to validate_submission.py.

    Returns:
        (exit_code, output_text)
    """
    if validator_script is None:
        # Search candidate locations
        search_paths = [
            os.path.join(workspace_root, "Resources", "student_resource", "utils", "validate_submission.py"),
            os.path.join(repo_root, "Resources", "student_resource", "utils", "validate_submission.py"),
            os.path.join(repo_root, "utils", "validate_submission.py"),
            os.path.join(dataset_dir, "utils", "validate_submission.py"),
            os.path.join(dataset_dir, "person3", "utils", "validate_submission.py"),
            os.path.abspath("utils/validate_submission.py"),
        ]
        for p in search_paths:
            if os.path.isfile(p):
                validator_script = p
                break

    if not validator_script or not os.path.isfile(validator_script):
        raise FileNotFoundError(f"validate_submission.py not found in expected paths.")

    if test_dir is None:
        search_test_dirs = [
            os.path.join(dataset_dir, "test"),
            os.path.join(repo_root, "Resources", "student_resource", "dataset", "test"),
        ]
        for td in search_test_dirs:
            if os.path.isdir(td) and os.path.isfile(os.path.join(td, "test_source1.tsv")):
                test_dir = td
                break

    if not test_dir:
        raise FileNotFoundError("test directory with test_source1.tsv not found.")

    cmd = [
        sys.executable,
        validator_script,
        "--matching",
        matching_path,
        "--test-dir",
        test_dir,
    ]

    if candidate_path and os.path.isfile(candidate_path):
        cmd.extend(["--candidate", candidate_path])

    if check_ids:
        cmd.append("--check-ids")

    print("=" * 80)
    print("PERSON 3: RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    output = proc.stdout + "\n" + proc.stderr
    print(output.strip())
    print("=" * 80)
    if proc.returncode == 0:
        print(">>> RESULT: VALIDATION PASSED (Exit code 0). Safe for leaderboard submission.")
    else:
        print(f">>> RESULT: VALIDATION FAILED (Exit code {proc.returncode}). Issues must be fixed.")
    print("=" * 80)

    return proc.returncode, output


def main():
    parser = argparse.ArgumentParser(description="Person 3 Submission Validator")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default=None, help="Directory containing test_source1.tsv")
    parser.add_argument("--check-ids", action="store_true", help="Enable memory-heavy ID existence check")
    args = parser.parse_args()

    exit_code, _ = run_validator(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
