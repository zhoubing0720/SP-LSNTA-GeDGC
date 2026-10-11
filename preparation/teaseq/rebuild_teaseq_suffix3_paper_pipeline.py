from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import anndata as ad
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp


EXPECTED = {
    "cells": 6310,
    "classes": 11,
    "rna_features": 9855,
    "atac_features": 14732,
    "adt_features": 46,
}

DNA_RE = re.compile(r"([ACGTN]{12,32})", re.IGNORECASE)
SUFFIX_RE = re.compile(r"-([0-9]+)$")


def dna_barcode(value: object) -> str:
    text = str(value).strip().strip('"').strip("'").upper()
    match = DNA_RE.search(text)
    if match:
        return match.group(1)
    return re.sub(r"-[0-9]+$", "", text)


def barcode_suffix(value: object) -> str | None:
    text = str(value).strip().strip('"').strip("'")
    match = SUFFIX_RE.search(text)
    return match.group(1) if match else None


def locate_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} under {root}; "
            f"found: {matches}"
        )
    return matches[0]


def unique_index(values) -> dict[str, int]:
    buckets: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        key = dna_barcode(value)
        buckets.setdefault(key, []).append(index)

    duplicates = {key: rows for key, rows in buckets.items() if len(rows) > 1}
    if duplicates:
        preview = list(duplicates.items())[:5]
        raise ValueError(
            "Duplicate DNA barcodes were found within one modality. "
            f"Examples: {preview}"
        )
    return {key: rows[0] for key, rows in buckets.items()}


def load_adt(path: Path) -> ad.AnnData:
    frame = pd.read_csv(path)
    barcode_candidates = [
        column for column in frame.columns
        if "barcode" in str(column).lower()
    ]
    if not barcode_candidates:
        raise KeyError(
            f"No barcode column found in ADT file. Columns: {list(frame.columns)}"
        )

    barcode_column = barcode_candidates[0]
    frame = frame.set_index(barcode_column)

    drop_columns = [
        column for column in frame.columns
        if str(column).strip().lower() in {"total", "total_counts"}
    ]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Non-numeric ADT columns detected: {bad}")

    return ad.AnnData(
        X=sp.csr_matrix(numeric.to_numpy(dtype=np.float32)),
        obs=pd.DataFrame(index=frame.index.astype(str)),
        var=pd.DataFrame(index=[f"adt:{column}" for column in numeric.columns]),
    )


def subset_by_dna(
    adata: ad.AnnData,
    ordered_dna: list[str],
    index_map: dict[str, int],
) -> ad.AnnData:
    indices = np.asarray([index_map[key] for key in ordered_dna], dtype=int)
    result = adata[indices].copy()
    result.obs_names = ordered_dna
    result.var_names_make_unique()
    return result


def filter_detected_in_at_least_one_percent(
    adata: ad.AnnData,
) -> tuple[ad.AnnData, int]:
    threshold = int(math.ceil(0.01 * adata.n_obs))
    matrix = adata.X
    if sp.issparse(matrix):
        detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    else:
        detected = np.count_nonzero(np.asarray(matrix) > 0, axis=0)

    keep = detected >= threshold
    return adata[:, keep].copy(), threshold


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the TEA-seq suffix-3 dataset without the extra strict "
            "cell-level QC that reduced the previous source file to 5805 cells."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("downloaded_data/teaseq"),
    )
    parser.add_argument(
        "--official-labels",
        type=Path,
        default=Path(
            "downloaded_data/teaseq/official_labels/extracted/"
            "Figure4_SourceData2_TypeLabelsUMAP.csv"
        ),
    )
    parser.add_argument("--suffix", default="3")
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("data/TEA_seq_suffix3_reconstructed.h5mu"),
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=Path("data/TEA_seq.h5mu"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "result/dataset_preparation/"
            "teaseq_suffix3_reconstruction_report.json"
        ),
    )
    args = parser.parse_args()

    h5_path = locate_one(args.raw_dir, "*filtered_feature_bc_matrix.h5")
    adt_path = locate_one(args.raw_dir, "*adt_counts.csv.gz")

    raw = mu.read_10x_h5(h5_path)
    if "rna" not in raw.mod or "atac" not in raw.mod:
        raise KeyError(
            f"Expected rna and atac modalities; found {list(raw.mod.keys())}"
        )

    rna = raw.mod["rna"].copy()
    atac = raw.mod["atac"].copy()
    adt = load_adt(adt_path)

    rna.var_names_make_unique()
    atac.var_names_make_unique()
    adt.var_names_make_unique()

    official = pd.read_csv(args.official_labels)
    if len(official.columns) < 4:
        raise ValueError(
            f"Official label table has too few columns: {list(official.columns)}"
        )

    # Verified official layout:
    # 0 barcode, 1 fine label, 3 broad Seurat label.
    barcode_column = official.columns[0]
    fine_column = official.columns[1]
    broad_column = official.columns[3]

    official = official.copy()
    official["batch_suffix"] = official[barcode_column].map(barcode_suffix)
    official["dna_barcode"] = official[barcode_column].map(dna_barcode)
    official = official.loc[
        official["batch_suffix"].astype(str) == str(args.suffix)
    ].copy()

    if official.empty:
        raise ValueError(f"No official rows found for suffix {args.suffix!r}")

    if official["dna_barcode"].duplicated().any():
        duplicates = official.loc[
            official["dna_barcode"].duplicated(keep=False),
            ["dna_barcode", barcode_column],
        ]
        raise ValueError(
            "Duplicate DNA barcodes exist inside the selected official batch.\n"
            + duplicates.head(20).to_string(index=False)
        )

    rna_index = unique_index(rna.obs_names)
    atac_index = unique_index(atac.obs_names)
    adt_index = unique_index(adt.obs_names)
    label_index = {
        value: index
        for index, value in official["dna_barcode"].items()
    }

    # Preserve official suffix-3 order. No random truncation is performed.
    common_dna = [
        dna
        for dna in official["dna_barcode"].tolist()
        if dna in rna_index
        and dna in atac_index
        and dna in adt_index
    ]

    rna = subset_by_dna(rna, common_dna, rna_index)
    atac = subset_by_dna(atac, common_dna, atac_index)
    adt = subset_by_dna(adt, common_dna, adt_index)

    label_rows = official.loc[
        [label_index[dna] for dna in common_dna]
    ].copy()
    broad_labels = label_rows[broad_column].astype(str).to_numpy()
    fine_labels = label_rows[fine_column].astype(str).to_numpy()

    # GeDGC supplement: remove features detected in fewer than 1% of cells.
    rna, rna_threshold = filter_detected_in_at_least_one_percent(rna)
    atac, atac_threshold = filter_detected_in_at_least_one_percent(atac)

    if adt.n_vars != EXPECTED["adt_features"]:
        raise ValueError(
            f"ADT features={adt.n_vars}, expected={EXPECTED['adt_features']}"
        )

    for modality in (rna, atac, adt):
        modality.obs["cell_type"] = broad_labels
        modality.obs["fine_cell_type"] = fine_labels

    mdata = md.MuData({"rna": rna, "atac": atac, "adt": adt})
    mdata.obs_names = common_dna
    mdata.obs["cell_type"] = broad_labels
    mdata.obs["fine_cell_type"] = fine_labels

    class_counts = (
        pd.Series(broad_labels)
        .value_counts()
        .sort_index()
    )

    actual = {
        "cells": int(mdata.n_obs),
        "classes": int(class_counts.size),
        "rna_features": int(rna.n_vars),
        "atac_features": int(atac.n_vars),
        "adt_features": int(adt.n_vars),
    }
    exact_match = actual == EXPECTED

    mdata.uns["gedgc_dataset_spec_expected"] = EXPECTED
    mdata.uns["gedgc_dataset_spec_actual"] = actual
    mdata.uns["gedgc_preparation"] = {
        "official_batch_suffix": str(args.suffix),
        "cell_selection": (
            "intersection of official suffix-3 labels, RNA, ATAC and ADT"
        ),
        "cell_level_qc": "none beyond existing filtered source matrices",
        "feature_filter": "detected in at least 1% of retained cells",
        "rna_detection_threshold_cells": int(rna_threshold),
        "atac_detection_threshold_cells": int(atac_threshold),
        "exact_paper_spec_match": bool(exact_match),
        "no_random_truncation": True,
        "no_manual_class_merging": True,
    }

    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    mdata.write(args.candidate_output)

    if exact_match:
        args.final_output.parent.mkdir(parents=True, exist_ok=True)
        mdata.write(args.final_output)

    report = {
        "raw_10x_h5": str(h5_path),
        "adt_file": str(adt_path),
        "official_labels": str(args.official_labels),
        "official_suffix": str(args.suffix),
        "official_suffix_rows": int(len(official)),
        "actual": actual,
        "expected": EXPECTED,
        "exact_paper_spec_match": bool(exact_match),
        "class_counts": {
            str(key): int(value) for key, value in class_counts.items()
        },
        "candidate_output": str(args.candidate_output),
        "final_output_written": str(args.final_output) if exact_match else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("TEA-SEQ SUFFIX-3 RECONSTRUCTION COMPLETE")
    print("=" * 88)
    print("Official suffix-3 rows:", len(official))
    print("Three-modality + label intersection:", mdata.n_obs)
    print("Classes:", class_counts.size)
    print("RNA features after 1% filter:", rna.n_vars)
    print("ATAC features after 1% filter:", atac.n_vars)
    print("ADT features:", adt.n_vars)
    print("\nClass counts:")
    print(class_counts.to_string())
    print("\nCandidate output:", args.candidate_output)
    print("Exact GeDGC table specification:", exact_match)

    if exact_match:
        print("FINAL OUTPUT WRITTEN:", args.final_output)
    else:
        print(
            "Final TEA_seq.h5mu was NOT overwritten because the exact "
            "6310/11/9855/14732/46 specification was not reproduced."
        )

    print("Report:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
