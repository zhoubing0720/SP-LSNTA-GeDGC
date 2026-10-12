# SP-LSNTA-GeDGC

Source code and processed inputs for **SP-LSNTA-GeDGC: Task-Oriented Shared-Private Routing and Multilevel Structural Alignment for Single-Cell Multi-Omics Clustering**, submitted to **Algorithms (MDPI)**.

Authors: Shizheng Zhang, Bing Zhou, Yanbu Guo, Min Huang, Wanwei Huang, and Dongyuan Lei. See CITATION.cff, LICENSE_NOTICE.md and THIRD_PARTY_NOTICES.md. Original data access conditions and third-party attribution remain applicable.

## Shared materials

This repository provides model source, execution configurations, data preparation and analysis code, processed input files and dataset documentation. Experimental results, figures, supplementary result tables and reviewer-response letters are not distributed here. Code execution generates outputs locally.

[Dataset construction, pairing, annotation use and molecular-view definitions](docs/DATASET_CONSTRUCTION.md) are explained in the repository. [Processed input release](https://github.com/zhoubing0720/SP-LSNTA-GeDGC/releases/tag/v1.0.0) supplies five data assets and two checksum manifests. Four smaller exact inputs remain in data/processed. See data/README.md and data/revision_release_status.json.

## Obtain and verify inputs

```text
python scripts/download_processed_revision.py --all
python scripts/verify_data.py --scope all
```

The legacy Kidney file preserves the exact numeric input with the documented old feature-header offset. A separately versioned annotation-corrected file has identical numeric matrices, cells and labels. Dataset dimensions, source conditions and reconstruction limits are described in the data documentation.

## Run locally

The full-pipeline source and launchers are reproduction/frozen_source and reproduction/runner. reproduction/protocol_template.json defines the data identities, seeds and component configurations. Use the scientific dependencies in reproduction/environment_versions.json; the documented environment is Python 3.8.20, PyTorch 2.1.2/CUDA 11.8 and an NVIDIA GPU.

```text
python reproduction/prepare_reproduction.py --data-dir data/processed --output new_runs
python new_runs/run_queue.py
python reproduction/analyze_all_runs.py --batch new_runs --out local_analysis
python reproduction/make_revision_figures.py --batch new_runs --analysis local_analysis
```

Preparation requires a new output directory and verifies input/source hashes. Each worker starts fresh pretraining and writes predictions, timing and memory records to the local output directory. See docs/REPRODUCIBILITY.md, reproduction/ANALYSIS_PLAN.md and docs/METRICS.md. The root run.py is a separate legacy convenience interface with a historical checkpoint-reuse option.

## Implemented method

VAE pretraining minimises reconstruction alone. A classifier is pretrained using GMM pseudo-labels. FG-MNN/UOT update graph sample centres, which become reconstruction targets in joint training. Structure is formed after pretraining and then fixed. Clustering K and shared latent dimension use the number of annotated categories, so evaluation is conditional on known K. Exact spectral/neighbour operations retain quadratic costs.

The feature gate uses a nonzero ratio after per-cell centering/scaling; this ratio is not raw detection frequency. This definition and the original input access conditions remain explicit without publishing experiment findings.
