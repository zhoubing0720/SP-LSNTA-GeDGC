# Data availability and preparation

The repository is designed to track only the following four selected processed
H5MU files after they are added by the release maintainer:

| Dataset | Required path | Size | SHA-256 |
| --- | --- | ---: | --- |
| Chen-2019 | `data/processed/Chen-2019.h5mu` | 20,007,217 bytes | `BDA2BB87F6D58B11D599E226D391CBAE73A9E21610FB03086DE7155DE4D7CE70` |
| Kidney | `data/processed/Kidney.h5mu` | 29,221,980 bytes | `C095BEF25A489E7A9D7462827225A91A027004C9AB276E3C309F375DAFC837C9` |
| multiome | `data/processed/multiome.h5mu` | 19,479,594 bytes | `BEF40BB5875A9F901AF7FDC8BCC6B6730DD788AB6DEFD326C1C12B1FD3920B30` |
| dyngen_branching | `data/processed/dyngen_branching.h5mu` | 14,053,112 bytes | `C02241529FE3683027FB56C3DC824EE66F24527E13483F4DED3BD3CDED112D89` |

Verify the local files before running experiments:

```bash
python scripts/verify_data.py
```

## Large processed files not stored in GitHub

The following paper-ready files are not stored directly in this repository:

| Dataset | Required path | Size | Reason |
| --- | --- | ---: | --- |
| Pbmc10k | `data/processed/Pbmc10k.h5mu` | 106,211,103 bytes | Exceeds GitHub's regular-file limit |
| opmultiome | `data/processed/opmultiome.h5mu` | 568,210,479 bytes | Exceeds GitHub's regular-file limit |
| TEA-seq | `data/processed/TEA-seq.h5mu` | 228,183,519 bytes | Exceeds GitHub's regular-file limit |

Use `python scripts/download_public_data.py --list` to see the public upstream
sources. The script downloads public source materials only; it does not claim to
recreate the exact paper-ready H5MU files. The original feature-selection,
cell-alignment, label-assignment, and H5MU-construction scripts were not present
in the supplied experiment package. Do not substitute a differently processed
file for a manuscript result without documenting that change.

## Dataset provenance

- **Chen-2019:** SNARE-seq adult mouse brain data associated with GEO GSE126074.
- **Kidney:** The processed file has 4,753 cells, but an upstream accession and
  conversion protocol were not verified in the supplied materials. It is included
  only after the authors confirm that public redistribution is permitted.
- **Pbmc10k:** Derived from annotated 10x Multiome PBMC10k RNA and ATAC materials
  distributed by the GLUE project.
- **opmultiome** and **multiome:** Derived from the Open Problems BMMC resource
  associated with GEO GSE194122. They are distinct processed subsets.
- **TEA-seq:** Public processed materials are associated with GEO GSE158013. Raw
  human data are controlled under dbGaP study `phs002316.v1.p1`; this repository
  does not bypass controlled-access requirements.
- **dyngen_branching:** Synthetic dataset associated with dyngen. The stored
  modalities are `rna`, `premrna`, `mrna`, and `adt`.

The exact expected file names, dimensions, accessions, and checksums are listed
in `dataset_manifest.csv`.
