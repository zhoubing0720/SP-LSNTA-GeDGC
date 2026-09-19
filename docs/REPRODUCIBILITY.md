# Reproducibility guide

This document describes what is included in the public repository, how the
paper configuration is resolved, and which limits apply when comparing a new
run with the manuscript tables.

## 1. Environment

The reference environment is defined in `environment.yml` and uses Python
3.7.12, PyTorch 1.13.1, Muon 0.1.3, and Scanpy 1.9.3. Create it with:

```bash
conda env create -f environment.yml
conda activate sp-lsnta-gedgc
```

GPU execution is recommended. The implementation constructs dense spectral
distance and neighborhood matrices, so memory consumption grows quadratically
with the number of cells in these stages.

## 2. Data integrity

Four processed datasets are hosted in `data/processed/`. Before running the
code, verify their byte sizes and SHA-256 checksums against
`data/dataset_manifest.csv`:

```bash
python scripts/verify_data.py
```

Pbmc10k, opmultiome, and TEA-seq are not hosted because each paper-ready H5MU
file exceeds GitHub's 100 MiB single-file limit. If the exact local files are
available, place them under `data/processed/` and run:

```bash
python scripts/verify_data.py --scope all
```

The upstream-download helper lists or downloads public source materials. It
does not recreate the exact non-hosted paper-ready files:

```bash
python scripts/download_public_data.py --list
```

## 3. Fixed experiment configuration

`configs/experiment.json` is the authoritative public configuration for graph
construction, SP routing, pretraining, FG-MNN, UOT, and SpectralNet hidden
widths. `run.py` validates the file and passes its values to the corresponding
training components. The checked-in values reproduce the fixed settings used
for the manuscript; modifying them defines a different experiment.

Dataset names, modalities, preprocessing rules, label fields, canonical file
names, and historical aliases are defined in `configs/datasets.json`.

## 4. Independent runs

Run one dataset at a time. The manuscript reports five independent runs, so an
explicit five-seed invocation is recommended:

```bash
python run.py --dataset Chen-2019 --seeds 0 1 2 3 4
```

The same operation can be expressed as consecutive seeds:

```bash
python run.py --dataset Chen-2019 --seed 0 --repeats 5
```

When a complete trusted VAE/GMM/classifier checkpoint set is unavailable for a
seed, the entry point starts pretraining automatically. Seed-specific runs do
not silently reuse an unlabelled checkpoint set from another seed.

## 5. Outputs and metric scales

Each run writes to `results/<dataset>/seed_<seed>/`:

- `metrics.csv`: metrics for that seed;
- `run_config.json`: resolved data and experiment settings plus checkpoint reuse
  information; and
- training-specific latent or trajectory artifacts.

After all requested seeds finish, `run.py` writes `runs.csv` and `summary.csv`
under `results/<dataset>/`. Runtime ACC, NMI, ARI, Purity, FMI, and Score values
use the 0--1 scale; runtime is in seconds. Values in
`results/paper_results.csv` are percentages on the 0--100 scale.

## 6. Reproducibility limits

CUDA kernels, GPU models, numerical-library versions, and stochastic
optimization can produce small run-to-run or platform-dependent differences.
The repository therefore provides the exact fixed configuration, processed-file
checksums, declared seeds, and mean/standard-deviation outputs, rather than
claiming bit-for-bit equality across all hardware.

The three non-hosted paper-ready datasets cannot be reconstructed exactly from
the public upstream downloads alone with the files currently released. Results
for those datasets require the expected processed H5MU files listed in the
manifest.
