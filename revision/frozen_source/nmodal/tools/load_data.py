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


def _dense_float_tensor(x):
    if sparse.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)

    return torch.tensor(
        x,
        dtype=torch.float32,
    )


def _manual_clr(adata):
    x = adata.X

    if sparse.issparse(x):
        x = x.toarray()

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    log1p_x = np.log1p(
        np.maximum(x, 0.0)
    )

    geom = np.exp(
        log1p_x.mean(
            axis=1,
            keepdims=True,
        )
    )

    clr = np.log1p(
        np.maximum(x, 0.0)
        / (geom + 1e-12)
    )

    adata.X = sparse.csr_matrix(
        clr
    )


def _clr(adata):
    try:
        mu.prot.pp.clr(
            adata
        )
        print(
            "Protein preprocessing: mu.prot.pp.clr"
        )
    except Exception as exc:
        print(
            "Use manual CLR fallback:",
            repr(exc),
        )
        _manual_clr(
            adata
        )


def _resolve_teaseq_path():
    candidates = [
        Path("data/TEA_seq.h5mu"),
        Path("data/TEA_seq.h5mu.gz"),
        Path("data/TEA-seq.h5mu"),
        Path("data/TEA-seq.h5mu.gz"),
        Path("data/TEAseq.h5mu"),
        Path("data/TEAseq.h5mu.gz"),
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        Path("data").glob("*TEA*.h5mu*")
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Cannot uniquely resolve TEA-seq file. "
        "Found: {}".format(
            [str(p) for p in matches]
        )
    )


def _resolve_dyngen_path():
    candidates = [
        Path(
            "data/dyngen_branching_compatible_13class.h5mu"
        ),
        Path(
            "data/dyngen_branching_compatible_13class.h5mu.gz"
        ),
        Path(
            "data/dyngen_branching.h5mu"
        ),
        Path(
            "data/dyngen_branching.h5mu.gz"
        ),
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        Path("data").glob(
            "*dyngen*branching*.h5mu*"
        )
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Cannot uniquely resolve dyngen file. "
        "Found: {}".format(
            [str(p) for p in matches]
        )
    )


def _find_key(
    data,
    aliases,
):
    keys = list(
        data.mod.keys()
    )

    lower_map = {
        str(key).lower(): key
        for key in keys
    }

    for alias in aliases:
        low = alias.lower()

        if low in lower_map:
            return lower_map[low]

    for key in keys:
        low = str(key).lower()

        if any(
            alias.lower() in low
            for alias in aliases
        ):
            return key

    return None


def _get_labels(rna_adata, data=None):
    preferred = [
        "cell_type",
        "celltype",
        "cell_type_l1",
        "celltype.l1",
        "label",
    ]

    for column in preferred:
        if column in rna_adata.obs.columns:
            print(
                "Label column:",
                column,
            )

            return (
                rna_adata.obs[column]
                .astype("category")
                .cat.codes
                .to_numpy()
            )

    if data is not None:
        for column in preferred:
            if column in data.obs.columns:
                print(
                    "Label column from MuData:",
                    column,
                )

                return (
                    data.obs[column]
                    .astype("category")
                    .cat.codes
                    .to_numpy()
                )

    raise KeyError(
        "No supported label column found."
    )


def _load_teaseq():
    path = _resolve_teaseq_path()

    print(
        "TEA-seq path:",
        path,
    )

    data = mu.read_h5mu(
        str(path)
    )

    print(
        "Available modalities:",
        list(data.mod.keys()),
    )

    rna_key = _find_key(
        data,
        [
            "rna",
            "gex",
            "gene_expression",
        ],
    )

    atac_key = _find_key(
        data,
        [
            "atac",
            "chromatin",
            "accessibility",
        ],
    )

    adt_key = _find_key(
        data,
        [
            "adt",
            "protein",
            "prot",
            "antibody",
        ],
    )

    if rna_key is None:
        raise KeyError(
            "RNA modality not found."
        )

    remaining = [
        key
        for key in data.mod.keys()
        if key != rna_key
    ]

    if adt_key is None and len(
        remaining
    ) >= 2:
        adt_key = min(
            remaining,
            key=lambda key: int(
                data[key].n_vars
            ),
        )

    if atac_key is None:
        candidate = [
            key
            for key in remaining
            if key != adt_key
        ]

        if len(candidate) == 1:
            atac_key = candidate[0]

    if (
        rna_key is None
        or atac_key is None
        or adt_key is None
        or len(
            {
                rna_key,
                atac_key,
                adt_key,
            }
        ) != 3
    ):
        raise KeyError(
            "Could not uniquely identify RNA/ATAC/ADT. "
            "Available: {}".format(
                list(data.mod.keys())
            )
        )

    print(
        "TEA-seq mapping:",
        rna_key,
        atac_key,
        adt_key,
    )

    sc.pp.normalize_total(
        data[rna_key],
        target_sum=1e4,
    )
    sc.pp.log1p(
        data[rna_key]
    )

    mu.atac.pp.tfidf(
        data[atac_key]
    )

    _clr(
        data[adt_key]
    )

    ordered = [
        rna_key,
        atac_key,
        adt_key,
    ]

    X = [
        normalization(
            _dense_float_tensor(
                data[key].X
            )
        )
        for key in ordered
    ]

    label = _get_labels(
        data[rna_key],
        data=data,
    )

    return (
        X,
        label,
        len(pd.unique(label)),
    )


def _load_dyngen():
    path = _resolve_dyngen_path()

    print(
        "dyngen path:",
        path,
    )

    data = mu.read_h5mu(
        str(path)
    )

    print(
        "Available modalities:",
        list(data.mod.keys()),
    )

    rna_key = _find_key(
        data,
        ["rna"],
    )

    premrna_key = _find_key(
        data,
        [
            "premrna",
            "pre-mrna",
            "pre_mrna",
        ],
    )

    mrna_key = _find_key(
        data,
        ["mrna"],
    )

    adt_key = _find_key(
        data,
        [
            "adt",
            "protein",
            "prot",
        ],
    )

    keys = [
        rna_key,
        premrna_key,
        mrna_key,
        adt_key,
    ]

    if (
        any(
            key is None
            for key in keys
        )
        or len(set(keys)) != 4
    ):
        raise KeyError(
            "Could not uniquely identify "
            "RNA/pre-mRNA/mRNA/ADT. "
            "Available: {}".format(
                list(data.mod.keys())
            )
        )

    print(
        "dyngen mapping:",
        keys,
    )

    for key in [
        rna_key,
        premrna_key,
        mrna_key,
    ]:
        sc.pp.normalize_total(
            data[key],
            target_sum=1e4,
        )
        sc.pp.log1p(
            data[key]
        )

    _clr(
        data[adt_key]
    )

    X = [
        normalization(
            _dense_float_tensor(
                data[key].X
            )
        )
        for key in keys
    ]

    label = _get_labels(
        data[rna_key],
        data=data,
    )

    return (
        X,
        label,
        len(pd.unique(label)),
    )


def load_data(dataset_name):
    normalized = (
        dataset_name
        .lower()
        .replace("-", "_")
    )

    if normalized in {
        "tea_seq",
        "teaseq",
    }:
        X, label, c = _load_teaseq()

    elif normalized in {
        "dyngen_branching",
        "dyngen",
    }:
        X, label, c = _load_dyngen()

    else:
        raise ValueError(
            "Final N-modal package supports "
            "TEA_seq and dyngen_branching. "
            "Got: {}".format(
                dataset_name
            )
        )

    n_cells = [
        int(x.shape[0])
        for x in X
    ]

    if len(
        set(n_cells)
    ) != 1:
        raise ValueError(
            "Modalities are not cell-aligned: {}".format(
                n_cells
            )
        )

    print(
        "Loaded N-modal shapes:",
        [tuple(x.shape) for x in X],
    )
    print(
        "Cells={}, modalities={}, classes={}".format(
            len(label),
            len(X),
            c,
        )
    )

    return X, label, c


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
