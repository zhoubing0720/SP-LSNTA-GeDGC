import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_support import (SEED_LIST, setup_seed, load_bimodal, begin_run,
                                ensure_pretrained, finish_run, cli)
from tools import Form_data
import torch
from train import train_opt
from pretrain import pretrain_opt
from graph_embedding.spectralnet import SpectralNet


def run_one_dataset(base_dataset, seed):
    info, completed = begin_run('GeDGC', base_dataset, seed)
    if completed is not None:
        return completed
    raw_data, label, c = load_bimodal(base_dataset)
    dataname = base_dataset + '/seed_' + str(seed)
    os.makedirs('result/' + dataname, exist_ok=True)
    start = time.perf_counter()
    pretrained = ensure_pretrained(pretrain_opt, raw_data, label, c, dataname, info)
    setup_seed(seed)
    spectralNet = SpectralNet(n_clusters=c, should_use_ae=True, should_use_siamese=True,
                             ae_hiddens=[512, c], siamese_hiddens=[256, c], spectral_hiddens=[256, c])
    data_cat = torch.cat(raw_data, dim=1)
    spectralNet.fit(data_cat)
    spectralNet_dist = spectralNet.predict(data_cat)
    torch.save(spectralNet_dist, 'Intermediate_data/' + dataname + '/SpectralNet_D.pkl')
    neighbors = Form_data.cal_neighbors_D(spectralNet_dist, 3)
    data = Form_data.form_data(raw_data, label, neighbors)
    setup_seed(seed)
    scores = train_opt(data, label, dataname)
    return finish_run('GeDGC', base_dataset, seed, scores,
                      time.perf_counter() - start, info, pretrained)


if __name__ == '__main__':
    cli('GeDGC', run_one_dataset)
