# SP-LSNTA-GeDGC

Source code, experiment configurations, verification utilities, and selected
processed datasets for *SP-LSNTA-GeDGC: Task-Oriented Shared-Private Routing and
Multilevel Structural Alignment for Single-Cell Multi-Omics Clustering*.

The associated manuscript is being prepared for submission to *Cells*. Please
cite the associated manuscript and this software using [`CITATION.cff`](CITATION.cff).

**Authors:** Shizheng Zhang, Bing Zhou, Yanbu Guo, Min Huang, Wanwei Huang, and
Dongyuan Lei. **Corresponding author:** Bing Zhou
([15290957496@163.com](mailto:15290957496@163.com)).

## Overview

SP-LSNTA-GeDGC is a graph-embedded deep generative clustering framework for
single-cell multi-omics integration. It combines:

- **SP routing**, which uses the shared latent representation for clustering and
  deterministic modality-specific residuals for reconstruction; and
- **LSNTA**, which combines shared-latent spectral structure, feature-gated
  mutual nearest neighbors, and unbalanced optimal transport.

![Framework overview](docs/model.png)

## Quick start

```bash
git clone https://github.com/zhoubing0720/SP-LSNTA-GeDGC-Cells.git
cd SP-LSNTA-GeDGC-Cells
conda env create -f environment.yml
conda activate sp-lsnta-gedgc

# Confirm the four hosted processed datasets are complete and unmodified.
python scripts/verify_data.py

# Check available datasets and the resolved paths without starting training.
python run.py --list-datasets
python run.py --dataset Chen-2019 --dry-run

# Run one dataset. Missing complete checkpoint sets trigger pretraining.
python run.py --dataset Chen-2019 --seed 0
```

The implementation was prepared for GPU execution. CPU execution is supported,
but can be impractical because dense spectral-distance and neighborhood matrices
are constructed. A dependency-only installation is also available through
`python -m pip install -r requirements.txt`.

## Data availability

Four paper-ready processed H5MU files are included in
[`data/processed/`](data/processed/) and are checked by file size and SHA-256:

| Dataset | Canonical file | Repository status |
| --- | --- | --- |
| Chen-2019 | `Chen-2019.h5mu` | Included |
| Kidney | `Kidney.h5mu` | Included |
| multiome | `multiome.h5mu` | Included |
| dyngen_branching | `dyngen_branching.h5mu` | Included |
| Pbmc10k | `Pbmc10k.h5mu` | Not hosted (>100 MiB) |
| opmultiome | `opmultiome.h5mu` | Not hosted (>100 MiB) |
| TEA-seq | `TEA-seq.h5mu` | Not hosted (>100 MiB; original access conditions apply) |

The file names, checksums, dimensions, and source records are listed in
[`data/dataset_manifest.csv`](data/dataset_manifest.csv). See
[`data/README.md`](data/README.md) for upstream sources and important limits on
reconstructing the three non-hosted paper-ready H5MU files.

```bash
# Verify the four datasets included in this repository.
python scripts/verify_data.py

# Verify all seven only after manually placing the three non-hosted files in
# data/processed/.
python scripts/verify_data.py --scope all
```

The `.gitignore` intentionally permits only the four hosted canonical H5MU
files. Do not add the three larger files to Git history.

## Reproducing an experiment

Run one dataset at a time. For the five independent runs described in the
manuscript, use a declared set of seeds:

```bash
python run.py --dataset Chen-2019 --seeds 0 1 2 3 4
```

`run.py` stores per-seed metrics under `results/<dataset>/seed_<seed>/`, writes a
dataset-level `runs.csv` and `summary.csv`, and saves the resolved configuration
for each run. It reuses a complete trusted checkpoint set when present; otherwise
it starts VAE, GMM, and classifier pretraining automatically. Generated
checkpoints, logs, and results are excluded from version control.

Useful release checks:

```bash
python run.py --dataset TEA-seq --dry-run
python -m unittest discover -s tests -v
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the expected
environment, data checks, output layout, and reproducibility limits.

## Repository layout

```text
.
├── configs/                 # Dataset registry and fixed experiment settings
├── data/                    # Hosted data, manifest, and source notes
├── docs/                    # Framework figure and reproducibility notes
├── results/                 # Manuscript table and generated-run location
├── scripts/                 # Source listing/download and data verification
├── src/                     # Model implementation
├── tests/                   # Lightweight release checks
├── run.py                   # One-dataset-at-a-time entry point
└── CITATION.cff
```

## Results

[`results/paper_results.csv`](results/paper_results.csv) records the GeDGC and
SP-LSNTA-GeDGC values reported in the manuscript. These values are expressed as
**percentages** and are not newly generated outputs. By contrast, metrics written
by `run.py` are proportions in the interval [0, 1] (except `Time`, in seconds);
multiply clustering metrics by 100 before comparing them with the manuscript
table. See [`results/README.md`](results/README.md).

## License and third-party material

No explicit software license was supplied with the original experiment package.
The repository therefore retains its all-rights-reserved notice until the authors
approve an explicit open-source license. See [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Public dataset sources
remain subject to their own terms and access conditions.

## Data and code availability statement for the manuscript

> The source code, configuration files, data-verification utilities, and
> experiment scripts used in this study are publicly accessible at
> https://github.com/zhoubing0720/SP-LSNTA-GeDGC-Cells. The processed
> Chen-2019, Kidney, multiome, and dyngen_branching datasets are available in
> the repository. The processed Pbmc10k, opmultiome, and TEA-seq datasets are
> not hosted because they exceed GitHub's single-file size limit and remain
> subject to their original access or redistribution conditions. Their original
> sources and preprocessing information are documented in this repository.
