"""Dataset loading with the preprocessing retained from the experiments."""

from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from torch.utils.data import Dataset


def normalization(input_):
    sample_mean = torch.mean(input_, dim=1).view(input_.shape[0], 1)
    sample_std = torch.std(input_, dim=1).view(input_.shape[0], 1)
    return (input_ - sample_mean) / (sample_std + 1e-10)


def _dense_float_tensor(x):
    if sparse.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)
    return torch.tensor(x, dtype=torch.float32)


def _manual_clr(adata):
    x = adata.X
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    log1p_x = np.log1p(np.maximum(x, 0.0))
    geom = np.exp(log1p_x.mean(axis=1, keepdims=True))
    adata.X = sparse.csr_matrix(
        np.log1p(np.maximum(x, 0.0) / (geom + 1e-12))
    )


def _clr(adata):
    try:
        mu.prot.pp.clr(adata)
        print("Protein preprocessing: mu.prot.pp.clr")
    except Exception as exc:
        print("Use manual CLR fallback:", repr(exc))
        _manual_clr(adata)


def _preprocess(adata, method):
    if method == "rna_log1p":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif method == "atac_tfidf":
        mu.atac.pp.tfidf(adata)
    elif method == "adt_clr":
        _clr(adata)
    else:
        raise ValueError("Unknown preprocessing method: {}".format(method))


def load_dataset(spec, data_dir):
    candidates = [spec["filename"]] + spec.get("legacy_filenames", [])
    existing = [Path(data_dir) / name for name in candidates if (Path(data_dir) / name).is_file()]
    if not existing:
        raise FileNotFoundError(
            "Required processed dataset is missing. Tried: {}. See data/README.md.".format(
                [str(Path(data_dir) / name) for name in candidates]
            )
        )
    path = existing[0]
    print("Data path:", path)
    data = mu.read_h5mu(str(path))
    expected = [item["name"] for item in spec["modalities"]]
    available = list(data.mod.keys())
    missing = [key for key in expected if key not in data.mod]
    if missing:
        raise KeyError(
            "Missing modalities {}. Available modalities: {}".format(missing, available)
        )

    matrices = []
    for item in spec["modalities"]:
        adata = data[item["name"]]
        _preprocess(adata, item["preprocessing"])
        matrices.append(normalization(_dense_float_tensor(adata.X)))

    label_modality = spec["label_modality"]
    label_column = spec["label_column"]
    if label_column not in data[label_modality].obs.columns:
        raise KeyError(
            "Label column {!r} is absent from modality {!r}. Available: {}".format(
                label_column,
                label_modality,
                list(data[label_modality].obs.columns),
            )
        )
    labels = (
        data[label_modality].obs[label_column]
        .astype("category")
        .cat.codes
        .to_numpy()
    )
    n_cells = [int(x.shape[0]) for x in matrices]
    if len(set(n_cells)) != 1:
        raise ValueError("Modalities are not cell-aligned: {}".format(n_cells))
    n_clusters = len(pd.unique(labels))
    print("Loaded shapes:", [tuple(x.shape) for x in matrices])
    print("Cells={}, modalities={}, classes={}".format(
        len(labels), len(matrices), n_clusters
    ))
    return matrices, labels, n_clusters


class Cell(Dataset):
    def __init__(self, data, labels=None, GMM_labels=None):
        self.num_views = len(data)
        self.data = data
        self.labels = labels
        self.GMM_labels = GMM_labels

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return [x[idx] for x in self.data]
        if self.GMM_labels is None:
            return [x[idx] for x in self.data], torch.from_numpy(self.labels)[idx]
        return (
            [x[idx] for x in self.data],
            torch.from_numpy(self.labels)[idx],
            self.GMM_labels[idx],
        )
