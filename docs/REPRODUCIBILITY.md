# Local reproduction

The versioned Release contains processed input data and checksums. Experimental outputs, scientific result figures and reviewer correspondence are kept outside this public repository.

The full-pipeline protocol is reproduction/protocol_template.json; scientific source and launchers are reproduction/frozen_source and reproduction/runner. The root run.py/configs/src interface is a separate legacy route. Dependencies are recorded in reproduction/environment_versions.json.

```text
python scripts/download_processed_revision.py --all
python scripts/verify_data.py --scope all
python reproduction/prepare_reproduction.py --data-dir data/processed --output new_runs
python new_runs/run_queue.py
python reproduction/analyze_all_runs.py --batch new_runs --out local_analysis
python reproduction/make_revision_figures.py --batch new_runs --analysis local_analysis
```

Choose a new output path. Preparation validates fingerprints and refuses overwrite; training starts fresh VAE/GMM/classifier, graph and joint optimization stages. Local outputs support independent metric recomputation, sample SD, paired confidence intervals and documented multiple-comparison families. The plan and metric scales describe procedures rather than published findings.

Posterior-mean exports can be visualized using preselected seed 0, all cells and common UMAP parameters. Metrics use sampled-latent classifier predictions. Runtime/GPU-memory measurements are generated locally with their documented scopes. Different hardware and library versions do not promise bitwise-identical training.

Dataset preparation, pairing, label use, reconstruction limits and conditions are in docs/DATASET_CONSTRUCTION.md and data/README.md.
