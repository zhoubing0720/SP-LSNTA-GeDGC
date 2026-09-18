#!/usr/bin/env python
"""Download public upstream source files; never accesses controlled data."""

import argparse
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = {
    "Chen-2019": [
        ("GSE126074_supplementary_files.tar", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126074&format=file")
    ],
    "Pbmc10k": [
        ("10x-Multiome-Pbmc10k-RNA.h5ad", "https://ftp.cbi.pku.edu.cn/pub/GLUE/dataset/10x-Multiome-Pbmc10k-RNA.h5ad"),
        ("10x-Multiome-Pbmc10k-ATAC.h5ad", "https://ftp.cbi.pku.edu.cn/pub/GLUE/dataset/10x-Multiome-Pbmc10k-ATAC.h5ad"),
    ],
    "opmultiome": [
        ("GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz")
    ],
    "multiome": [
        ("GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz")
    ],
    "TEA-seq": [
        ("GSE158013_public_supplementary_files.tar", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE158013&format=file")
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DOWNLOADS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        for dataset, entries in DOWNLOADS.items():
            print(dataset)
            for filename, url in entries:
                print("  {} <- {}".format(filename, url))
        print("Kidney and exact dyngen processed inputs: no verified automatic source.")
        print("TEA-seq raw reads: controlled dbGaP phs002316.v1.p1; not downloaded.")
        return 0
    if not args.dataset:
        raise SystemExit("--dataset is required unless --list is used")
    destination = args.output_dir / args.dataset
    destination.mkdir(parents=True, exist_ok=True)
    for filename, url in DOWNLOADS[args.dataset]:
        path = destination / filename
        if path.exists() and not args.overwrite:
            print("SKIP existing:", path)
            continue
        print("DOWNLOAD", url)
        with urllib.request.urlopen(url) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        print("SAVED", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
