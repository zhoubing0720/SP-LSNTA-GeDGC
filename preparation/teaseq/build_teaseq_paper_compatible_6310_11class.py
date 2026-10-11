from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp


DROP_RARE_CLASSES = ("DC.Plasmacytoid", "T.DoubleNegative")
TARGET_ATAC_FEATURES = 14732
EXPECTED = {
    "cells": 6310,
    "classes": 11,
    "rna_features": 9855,
    "atac_features": 14732,
    "adt_features": 46,
}


def sparse_tfidf_variance(x: sp.spmatrix) -> np.ndarray:
    """
    Compute a TF-IDF-based variance score for each ATAC feature without
    densifying the full cells × peaks matrix.

    TF = count / total counts per cell
    IDF = log(1 + n_cells / (1 + cells_detected))
    Score = variance of TF-IDF values across cells
    """
    x = x.tocsr().astype(np.float64)
    n_cells = x.shape[0]

    row_sum = np.asarray(x.sum(axis=1)).ravel()
    row_sum[row_sum <= 0] = 1.0
    inv_row_sum = 1.0 / row_sum

    tf = sp.diags(inv_row_sum) @ x
    detected = np.asarray((x > 0).sum(axis=0)).ravel()
    idf = np.log1p(n_cells / (1.0 + detected))

    tfidf = tf @ sp.diags(idf)

    mean = np.asarray(tfidf.mean(axis=0)).ravel()
    mean_sq = np.asarray(tfidf.power(2).mean(axis=0)).ravel()
    variance = mean_sq - mean * mean
    variance[~np.isfinite(variance)] = -np.inf
    return variance


def get_labels(mdata) -> pd.Series:
    if "cell_type" in mdata.obs:
        return mdata.obs["cell_type"].astype(str)
    for modality in mdata.mod.values():
        if "cell_type" in modality.obs:
            return modality.obs["cell_type"].astype(str)
    raise KeyError("No cell_type column found in MuData.obs or modality obs.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a scientifically labeled TEA-seq paper-shape-compatible "
            "dataset from the suffix-3 reconstruction."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TEA_seq_suffix3_reconstructed.h5mu"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/TEA_seq_paper_compatible_6310_11class.h5mu"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "result/dataset_preparation/"
            "teaseq_paper_compatible_report.json"
        ),
    )
    parser.add_argument(
        "--write-final-alias",
        action="store_true",
        help=(
            "Also write data/TEA_seq.h5mu after all five target dimensions "
            "match. This does not make the reconstruction an exact replica."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    source = mu.read_h5mu(str(args.input))
    labels = get_labels(source)

    before_counts = labels.value_counts().sort_index()
    missing_drop_classes = [
        label for label in DROP_RARE_CLASSES
        if label not in before_counts.index
    ]
    if missing_drop_classes:
        raise ValueError(
            f"Expected rare classes are absent: {missing_drop_classes}\n"
            f"Observed classes:\n{before_counts.to_string()}"
        )

    keep = ~labels.isin(DROP_RARE_CLASSES)
    retained_names = source.obs_names[keep.to_numpy()]

    modalities = {}
    for name, modality in source.mod.items():
        modality_keep = modality.obs_names.isin(retained_names)
        subset = modality[modality_keep].copy()
        # Reorder to the retained MuData order.
        position = pd.Series(
            np.arange(subset.n_obs), index=subset.obs_names
        )
        order = position.loc[retained_names].to_numpy()
        subset = subset[order].copy()
        subset.obs_names = retained_names
        modalities[name] = subset

    if "rna" not in modalities or "atac" not in modalities or "adt" not in modalities:
        raise KeyError(
            f"Expected rna/atac/adt modalities, found {list(modalities)}"
        )

    rna = modalities["rna"]
    atac = modalities["atac"]
    adt = modalities["adt"]

    if atac.n_vars < TARGET_ATAC_FEATURES:
        raise ValueError(
            f"ATAC has only {atac.n_vars} features; "
            f"cannot select {TARGET_ATAC_FEATURES}."
        )

    if atac.n_vars > TARGET_ATAC_FEATURES:
        x = atac.X
        if not sp.issparse(x):
            x = sp.csr_matrix(np.asarray(x))
        scores = sparse_tfidf_variance(x)
        # Stable descending selection; then sort indices to preserve original
        # feature order in the saved count matrix.
        selected = np.argsort(-scores, kind="stable")[:TARGET_ATAC_FEATURES]
        selected = np.sort(selected)
        atac = atac[:, selected].copy()

    final_labels = labels.loc[retained_names].astype(str)
    for modality in (rna, atac, adt):
        modality.obs["cell_type"] = final_labels.to_numpy()

    result = md.MuData({"rna": rna, "atac": atac, "adt": adt})
    result.obs_names = retained_names
    result.obs["cell_type"] = final_labels.to_numpy()

    after_counts = final_labels.value_counts().sort_index()
    actual = {
        "cells": int(result.n_obs),
        "classes": int(after_counts.size),
        "rna_features": int(rna.n_vars),
        "atac_features": int(atac.n_vars),
        "adt_features": int(adt.n_vars),
    }
    paper_shape_match = actual == EXPECTED

    result.uns["gedgc_dataset_spec_expected"] = EXPECTED
    result.uns["gedgc_dataset_spec_actual"] = actual
    result.uns["gedgc_preparation"] = {
        "source": str(args.input),
        "removed_rare_classes": list(DROP_RARE_CLASSES),
        "removed_cells": int((~keep).sum()),
        "atac_selection": (
            "top 14732 features by sparse TF-IDF variance; "
            "original count matrix retained for selected peaks"
        ),
        "paper_shape_match": bool(paper_shape_match),
        "exact_original_feature_identity_known": False,
        "scientific_status": (
            "paper-shape-compatible reconstruction, not an exact replica "
            "of the unpublished GeDGC preprocessing output"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write(args.output)

    final_alias = None
    if args.write_final_alias and paper_shape_match:
        final_alias = Path("data/TEA_seq.h5mu")
        result.write(final_alias)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "final_alias": str(final_alias) if final_alias else None,
        "before": {
            "cells": int(source.n_obs),
            "classes": int(before_counts.size),
            "class_counts": {
                str(key): int(value)
                for key, value in before_counts.items()
            },
        },
        "removed_classes": list(DROP_RARE_CLASSES),
        "removed_cells": int((~keep).sum()),
        "after": actual,
        "expected": EXPECTED,
        "paper_shape_match": bool(paper_shape_match),
        "class_counts": {
            str(key): int(value)
            for key, value in after_counts.items()
        },
        "scientific_note": (
            "The 6310 cells and 11 classes arise naturally by excluding the "
            "two rare classes containing 3 and 1 cells. The ATAC dimension is "
            "matched using TF-IDF variance selection because the exact peak "
            "identity used by GeDGC is not publicly available."
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("TEA-SEQ PAPER-SHAPE-COMPATIBLE DATASET COMPLETE")
    print("=" * 92)
    print("Input cells/classes:", source.n_obs, before_counts.size)
    print("Removed classes:", ", ".join(DROP_RARE_CLASSES))
    print("Removed cells:", int((~keep).sum()))
    print("Output cells:", result.n_obs)
    print("Output classes:", after_counts.size)
    print("RNA features:", rna.n_vars)
    print("ATAC features:", atac.n_vars)
    print("ADT features:", adt.n_vars)
    print("Paper shape match:", paper_shape_match)
    print("\nClass counts:")
    print(after_counts.to_string())
    print("\nOutput:", args.output)
    if final_alias:
        print("Final alias:", final_alias)
    print("Report:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
