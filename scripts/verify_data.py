#!/usr/bin/env python
"""Verify processed dataset presence, size, and SHA-256 against the manifest."""

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITHUB_DATASETS = {"Chen-2019", "Kidney", "multiome", "dyngen_branching"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("github", "all"),
        default="github",
        help="verify the four GitHub-hosted datasets (default) or all seven",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    args = parse_args()
    failed = False
    with (ROOT / "data" / "dataset_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if args.scope == "github" and row["dataset_name"] not in GITHUB_DATASETS:
                continue
            path = ROOT / row["expected_local_path"]
            if not path.is_file():
                print("MISSING", row["dataset_name"], path)
                failed = True
                continue
            actual = sha256(path)
            ok = actual == row["checksum"].upper()
            print("OK" if ok else "MISMATCH", row["dataset_name"], actual)
            failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
