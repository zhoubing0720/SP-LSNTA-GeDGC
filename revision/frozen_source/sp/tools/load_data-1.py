import pandas as pd
import torch
import muon as mu
import scanpy as sc
from torch.utils.data import Dataset


def normalization(input_):
    sampleMean = torch.mean(input_, dim=1).view(input_.shape[0], 1)
    sampleStd = torch.std(input_, dim=1).view(input_.shape[0], 1)

    input_ = (input_ - sampleMean) / (sampleStd + 1e-10)
    return input_

def load_data():
    data = mu.read_h5mu("data/brain_H3K27ac.h5mu")
    sc.pp.normalize_total(data["rna"], target_sum=1e4)
    sc.pp.log1p(data["rna"])
    mu.atac.pp.tfidf(data['atac'])
    # sc.pp.normalize_total(data["premRNA"], target_sum=1e4)
    # sc.pp.log1p(data["premRNA"])
    # sc.pp.normalize_total(data["mRNA"], target_sum=1e4)
    # sc.pp.log1p(data["mRNA"])
    # mu.prot.pp.clr(data["adt"])

    X = []
    X.append(normalization(torch.tensor(data['rna'].X.todense()).float()))
    X.append(normalization(torch.tensor(data['atac'].X.todense()).float()))
    # X.append(normalization(torch.tensor(data['premRNA'].X.todense()).float()))
    # X.append(normalization(torch.tensor(data['mRNA'].X.todense()).float()))
    # X.append(normalization(torch.tensor(data['adt'].X.todense()).float()))

    label = (data['rna'].obs['cell_type'].cat.codes).to_numpy()

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
