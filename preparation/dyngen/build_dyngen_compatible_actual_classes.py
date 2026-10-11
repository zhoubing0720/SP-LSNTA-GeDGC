from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy.io import mmread


MODALITIES = ("rna", "premrna", "mrna", "adt")
EXPECTED_CELLS = 1000
EXPECTED_FEATURES = 600


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed-seed compatible dyngen dataset without forcing "
            "the unpublished GeDGC class count."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("downloaded_data/dyngen_branching/exported"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dyngen_branching_compatible_13class.h5mu"),
    )
    args = parser.parse_args()

    barcodes = [
        line.strip()
        for line in (args.export_dir / "barcodes.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    features = [
        line.strip()
        for line in (args.export_dir / "features.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    metadata = pd.read_csv(args.export_dir / "cell_metadata.csv")

    required = {"from", "to"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(
            f"Missing metadata columns: {sorted(missing)}. "
            f"Available: {list(metadata.columns)}"
        )

    if len(barcodes) != EXPECTED_CELLS:
        raise ValueError(
            f"barcodes={len(barcodes)}, expected={EXPECTED_CELLS}"
        )
    if len(metadata) != EXPECTED_CELLS:
        raise ValueError(
            f"metadata rows={len(metadata)}, expected={EXPECTED_CELLS}"
        )
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"features={len(features)}, expected={EXPECTED_FEATURES}"
        )

    labels = (
        metadata["from"].astype(str).str.strip()
        + "->"
        + metadata["to"].astype(str).str.strip()
    )
    class_counts = labels.value_counts().sort_index()
    observed_classes = int(class_counts.size)

    modalities = {}
    for name in MODALITIES:
        matrix_path = args.export_dir / f"{name}.mtx"
        matrix = mmread(matrix_path).tocsr().astype(np.float32)
        expected_shape = (EXPECTED_CELLS, EXPECTED_FEATURES)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{name} shape={matrix.shape}, expected={expected_shape}"
            )

        modalities[name] = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(
                {
                    "cell_type": labels.to_numpy(),
                    "from": metadata["from"].astype(str).to_numpy(),
                    "to": metadata["to"].astype(str).to_numpy(),
                },
                index=barcodes,
            ),
            var=pd.DataFrame(
                index=[f"{name}:{feature}" for feature in features]
            ),
        )

    mdata = md.MuData(modalities)
    mdata.obs_names = barcodes
    mdata.obs["cell_type"] = labels.to_numpy()
    mdata.obs["from"] = metadata["from"].astype(str).to_numpy()
    mdata.obs["to"] = metadata["to"].astype(str).to_numpy()
    if "sim_time" in metadata.columns:
        mdata.obs["sim_time"] = metadata["sim_time"].to_numpy()

    mdata.uns["gedgc_dataset_spec"] = {
        "cells": EXPECTED_CELLS,
        "observed_classes": observed_classes,
        "features": {
            name: EXPECTED_FEATURES for name in MODALITIES
        },
    }
    mdata.uns["gedgc_preparation"] = {
        "source": "fixed-seed dyngen compatible simulation",
        "class_definition": "from + '->' + to",
        "exact_paper_replica": False,
        "paper_class_count_forced": False,
        "scientific_note": (
            "This compatible simulation produced its own transition classes. "
            "No transitions were arbitrarily merged to imitate an unpublished "
            "benchmark labeling rule."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mdata.write(args.output)

    print("=" * 72)
    print("DYNGEN COMPATIBLE H5MU WRITTEN")
    print("=" * 72)
    print("Output:", args.output)
    print("Cells:", mdata.n_obs)
    print("Classes:", observed_classes)
    print("Features per modality:", EXPECTED_FEATURES)
    print(class_counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
