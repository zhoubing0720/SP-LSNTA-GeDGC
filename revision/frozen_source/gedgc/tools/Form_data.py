import torch
import numpy as np
from tools.utils import cluster_acc

def to_numpy(x):
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()

def cal_neighbors_S(S, K):
    # S = S.cuda()
    ind = torch.argsort(S, dim=1, descending=True)
    neighbors = ind[:, 0:K + 1].view(-1, K + 1)
    return neighbors.cpu()

def cal_neighbors_D(D, K):
    n = D.shape[0]
    neighbors = torch.from_numpy(np.zeros((n, K + 1)))
    for idx in range(n):
        ind = torch.argsort(D[idx, :])
        neighbors[idx] = ind[0:K + 1].view(1, K + 1)
    neighbors[:, 0] = torch.arange(n)
    
    return neighbors.type(torch.long)

def form_data(data, label, neighbors):
    form_data = []

    for i, data_v in enumerate(data):
        K = neighbors.shape[1]

        form_data_v = []
        for idx in range(K):
            data_temp = data_v[neighbors[:, idx]].unsqueeze(2)
            form_data_v.append(data_temp)

        form_data.append(torch.cat(form_data_v, dim=-1))

    for idx in range(neighbors.shape[1]):
        acc, _ = cluster_acc(label[neighbors[:, idx]], label)
        print(acc)

    return form_data