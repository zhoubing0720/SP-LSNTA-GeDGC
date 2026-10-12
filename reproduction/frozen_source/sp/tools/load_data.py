from pathlib import Path

import numpy as np
import pandas as pd
import torch
import muon as mu
import scanpy as sc
from scipy import sparse
from torch.utils.data import Dataset


def normalization(input_):
    sampleMean = torch.mean(
        input_,
        dim=1,
    ).view(input_.shape[0], 1)

    sampleStd = torch.std(
        input_,
        dim=1,
    ).view(input_.shape[0], 1)

    return (
        input_ - sampleMean
    ) / (
        sampleStd + 1e-10
    )


def _resolve_path(dataset_name):
    candidates = [
        Path("data") / f"{dataset_name}.h5mu",
        Path("data") / f"{dataset_name}.h5mu.gz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset file not found for {}. Tried: {}".format(
            dataset_name,
            [str(p) for p in candidates],
        )
    )


def _dense_float_tensor(x):
    if sparse.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)

    return torch.tensor(
        x,
        dtype=torch.float32,
    )


def load_data(dataset_name):
    """
    Final bimodal loader for:
        Chen_high
        Kidney_4753
        Pbmc10k
        opmultiome
        multiome_high
    """

    path = _resolve_path(dataset_name)
    print("Data path:", path)

    data = mu.read_h5mu(str(path))

    if "rna" not in data.mod:
        raise KeyError(
            "RNA modality not found. Available modalities: {}".format(
                list(data.mod.keys())
            )
        )

    second_keys = [
        key
        for key in data.mod.keys()
        if key != "rna"
    ]

    if len(second_keys) != 1:
        raise KeyError(
            "Final bimodal version requires exactly 2 modalities. "
            "Available modalities: {}".format(
                list(data.mod.keys())
            )
        )

    second_key = second_keys[0]

    print(
        "Bimodal mapping: RNA=rna, second={}".format(
            second_key
        )
    )

    # RNA preprocessing.
    sc.pp.normalize_total(
        data["rna"],
        target_sum=1e4,
    )
    sc.pp.log1p(
        data["rna"]
    )

    # Second modality follows the retained GeDGC
    # RNA+ATAC/multiome preprocessing convention.
    mu.atac.pp.tfidf(
        data[second_key]
    )

    X = [
        normalization(
            _dense_float_tensor(
                data["rna"].X
            )
        ),
        normalization(
            _dense_float_tensor(
                data[second_key].X
            )
        ),
    ]

    if "cell_type" not in data["rna"].obs.columns:
        raise KeyError(
            "cell_type not found in RNA obs. Available columns: {}".format(
                list(data["rna"].obs.columns)
            )
        )

    label = (
        data["rna"]
        .obs["cell_type"]
        .astype("category")
        .cat.codes
        .to_numpy()
    )

    n_clusters = len(
        pd.unique(label)
    )

    print(
        "Loaded shapes:",
        [tuple(x.shape) for x in X],
    )
    print(
        "Cells={}, classes={}".format(
            len(label),
            n_clusters,
        )
    )

    return X, label, n_clusters


class Cell(Dataset):
    def __init__(
        self,
        data,
        labels=None,
        GMM_labels=None,
    ):
        self.num_views = len(data)
        self.data = data
        self.labels = labels
        self.GMM_labels = GMM_labels

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return [
                x[idx]
                for x in self.data
            ]

        elif self.GMM_labels is None:
            return (
                [
                    x[idx]
                    for x in self.data
                ],
                torch.from_numpy(
                    self.labels
                )[idx],
            )

        else:
            return (
                [
                    x[idx]
                    for x in self.data
                ],
                torch.from_numpy(
                    self.labels
                )[idx],
                self.GMM_labels[idx],
            )
