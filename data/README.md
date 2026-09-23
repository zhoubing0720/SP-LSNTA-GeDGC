# Data availability and preparation

This repository hosts the following four selected processed H5MU files. They are the exact files expected by the public entry point and can be verified by size and SHA-256:

| Dataset | Required path | Size | SHA-256 |
|---|---|---:|---|
| Chen-2019 | `data/processed/Chen-2019.h5mu` | 20,007,217 bytes | `BDA2BB87F6D58B11D599E226D391CBAE73A9E21610FB03086DE7155DE4D7CE70` |
| Kidney | `data/processed/Kidney.h5mu` | 29,221,980 bytes | `C095BEF25A489E7A9D7462827225A91A027004C9AB276E3C309F375DAFC837C9` |
| multiome | `data/processed/multiome.h5mu` | 19,479,594 bytes | `BEF40BB5875A9F901AF7FDC8BCC6B6730DD788AB6DEFD326C1C12B1FD3920B30` |
| dyngen_branching | `data/processed/dyngen_branching.h5mu` | 14,053,112 bytes | `C02241529FE3683027FB56C3DC824EE66F24527E13483F4DED3BD3CDED112D89` |

Verify the hosted files before running experiments:

```bash
python scripts/verify_data.py
```

## Processed files not hosted in GitHub

The following paper-ready files are not stored directly in this repository:

| Dataset | Required path | Size | Reason |
|---|---|---:|---|
| Pbmc10k | `data/processed/Pbmc10k.h5mu` | 106,211,103 bytes | Exceeds GitHub's 100 MiB single-file limit |
| opmultiome | `data/processed/opmultiome.h5mu` | 568,210,479 bytes | Exceeds GitHub's 100 MiB single-file limit |
| TEA-seq | `data/processed/TEA-seq.h5mu` | 228,183,519 bytes | Exceeds GitHub's 100 MiB single-file limit; original access conditions apply |

Use `python scripts/download_public_data.py --list` to see the public upstream sources. The script downloads public source materials only; it does not recreate the exact paper-ready H5MU files. The feature-selection, cell-alignment, label-assignment, and H5MU-construction scripts required for the three non-hosted files were not available in the released experiment package. Do not substitute a differently processed file for a manuscript result without documenting that change.

## Dataset provenance

- **Chen-2019:** SNARE-seq adult mouse brain data associated with GEO GSE126074.
- **Kidney:** Adult human kidney RNA/ATAC data associated with Muto et al. (2021), *Nature Communications*, 12, 2190, doi:10.1038/s41467-021-22368-w. The repository provides the processed 4,753-cell benchmark used in this study; the original conversion procedure is not reconstructed here.
- **Pbmc10k:** Derived from annotated 10x Multiome PBMC10k RNA and ATAC materials distributed by the GLUE project.
- **opmultiome and multiome:** Derived from the Open Problems BMMC resource associated with GEO GSE194122. They are distinct processed subsets.
- **TEA-seq:** Public processed materials are associated with GEO GSE158013. Raw human data are controlled under dbGaP study `phs002316.v1.p1`; this repository does not bypass controlled-access requirements.
- **dyngen_branching:** Synthetic dataset associated with dyngen. The stored modalities are `rna`, `premrna`, `mrna`, and `adt`.

The exact expected file names, dimensions, accessions, and checksums are listed in `dataset_manifest.csv`.

## Manuscript data availability statement

> The processed Chen-2019, Kidney, multiome, and dyngen_branching datasets used in this study are publicly available in the project repository at https://github.com/zhoubing0720/SP-LSNTA-GeDGC/tree/main/data/processed. The processed Pbmc10k, opmultiome, and TEA-seq datasets are not hosted in the repository because they exceed GitHub's single-file size limit and are subject to their original access or redistribution conditions.

The corresponding original data sources are cited in the manuscript, and source-access and preprocessing information is provided here.
