# Third-party notices

## Software dependencies

This repository declares runtime dependencies including PyTorch, Muon, Scanpy,
NumPy, pandas, SciPy, scikit-learn, tqdm, Annoy, Matplotlib, and Munkres. These
packages are not vendored as installation artifacts in this repository and are
distributed under their respective licenses.

## Bundled SpectralNet implementation

The Python files under `src/graph_embedding/spectralnet/` were present in the
supplied experiment package but did not contain a standalone provenance or
license notice. Their origin and redistribution terms must be confirmed by the
authors before the repository is assigned an open-source license or redistributed
under terms that cover those files.

## Data sources

Dataset metadata, public upstream links, and controlled-access restrictions are
listed in `data/dataset_manifest.csv` and `data/README.md`. No controlled-access
data are downloaded or redistributed. Users are responsible for complying with
the terms of each upstream data provider.
