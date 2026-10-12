import pandas as pd
import torch
import muon as mu
import scanpy as sc
import numpy as np
from torch.utils.data import Dataset


def normalization(input_):
    sampleMean = torch.mean(input_, dim=1).view(input_.shape[0], 1)
    sampleStd = torch.std(input_, dim=1).view(input_.shape[0], 1)
    input_ = (input_ - sampleMean) / (sampleStd + 1e-10)
    return input_


def _to_dense_float_tensor(x):
    """Convert sparse / numpy matrices to float32 torch.Tensor."""
    if hasattr(x, "todense"):
        x = np.asarray(x.todense())
    else:
        x = np.asarray(x)
    return torch.tensor(x, dtype=torch.float32)


def _clr_inplace(adata):
    """
    CLR transform for ADT.
    Prefer Muon's protein preprocessing; fall back to an equivalent
    per-cell CLR-like transformation if that API is unavailable.
    """
    try:
        mu.prot.pp.clr(adata)
        return
    except Exception:
        X = adata.X
        if hasattr(X, "todense"):
            X = np.asarray(X.todense(), dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        log1p_x = np.log1p(X)
        gm_log = np.mean(log1p_x, axis=1, keepdims=True)
        X = np.log1p(X / (np.exp(gm_log) + 1e-12))
        adata.X = X.astype(np.float32)


def load_data():
    data = mu.read_h5mu("data/dyngen_branching.h5mu")

    # The current compatible dyngen file contains four modalities:
    # RNA / pre-mRNA / mRNA / ADT
    required = ["rna", "premrna", "mrna", "adt"]
    missing = [m for m in required if m not in data.mod]
    if missing:
        raise KeyError(
            "Missing modality/modalities: "
            + ", ".join(missing)
            + ". Available modalities: "
            + ", ".join(map(str, data.mod.keys()))
        )

    # Count-like RNA modalities
    for mod in ["rna", "premrna", "mrna"]:
        sc.pp.normalize_total(data[mod], target_sum=1e4)
        sc.pp.log1p(data[mod])

    # Protein modality
    _clr_inplace(data["adt"])

    # Keep the same sample-wise normalization used by the current project.
    X = [
        normalization(_to_dense_float_tensor(data["rna"].X)),
        normalization(_to_dense_float_tensor(data["premrna"].X)),
        normalization(_to_dense_float_tensor(data["mrna"].X)),
        normalization(_to_dense_float_tensor(data["adt"].X)),
    ]

    # Keep label definition consistent with the current GeDGC code.
    if "cell_type" in data["rna"].obs.columns:
        cell_type = data["rna"].obs["cell_type"]
    elif "cell_type" in data.obs.columns:
        cell_type = data.obs["cell_type"]
    else:
        raise KeyError("No 'cell_type' annotation found in RNA obs or MuData obs.")

    try:
        label = cell_type.cat.codes.to_numpy()
    except Exception:
        label = pd.Categorical(cell_type).codes

    print("Loaded dyngen_branching modalities:")
    for name, tensor in zip(required, X):
        print("  {}: {}".format(name, tuple(tensor.shape)))
    print("Cells:", len(label))
    print("Clusters:", len(pd.unique(label)))

    return X, label, len(pd.unique(label))


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
        elif self.GMM_labels is None:
            return [x[idx] for x in self.data], torch.from_numpy(self.labels)[idx]
        else:
            return [x[idx] for x in self.data], torch.from_numpy(self.labels)[idx], self.GMM_labels[idx]
