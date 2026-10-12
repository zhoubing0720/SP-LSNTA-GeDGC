import os
import gc
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_support import (SEED_LIST, setup_seed, begin_run,
                                ensure_pretrained, finish_run, cli)

import torch
import numpy as np
from torch.utils.data import DataLoader

from tools.load_data import load_data, Cell
from tools import Form_data
from train import train_opt
from pretrain import pretrain_opt
from graph_embedding.spectralnet import SpectralNet


GRAPH_K = 3

# Run exactly ONE dataset each time.
DATASET_NAME = "Chen_high"
# Available datasets: Chen_high, Kidney_4753, Pbmc10k, opmultiome, multiome_high


def get_pretrained_shared_latent(
    raw_data,
    dataname,
    batch_size=1024,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vae = torch.load(
        "Intermediate_data/"
        + dataname
        + "/pretrained_vae.pkl"
    )
    vae = vae.to(device)
    vae.eval()

    loader = DataLoader(
        Cell(raw_data),
        batch_size=batch_size,
        shuffle=False,
    )

    latent_chunks = []

    with torch.no_grad():
        for inputs in loader:
            inputs = [
                value.to(device).float()
                for value in inputs
            ]

            mean, _ = vae.get_latent(inputs)
            latent_chunks.append(
                mean.detach().cpu()
            )

    z_shared = torch.cat(
        latent_chunks,
        dim=0,
    ).float()

    print(
        "Latent-SpectralNet z_shared shape:",
        tuple(z_shared.shape),
    )

    vae = vae.to("cpu")
    del vae
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return z_shared


def run_one_dataset(base_dataset, seed):
    info, completed = begin_run('SP-LSNTA-GeDGC', base_dataset, seed)
    if completed is not None:
        return completed

    raw_data, label, c = load_data(base_dataset)

    dataname = (
        base_dataset
        + "_FINAL_SP_LSNTA_GeDGC"
        + "/seed_" + str(seed)
    )

    os.makedirs(
        "Intermediate_data/" + dataname,
        exist_ok=True,
    )
    os.makedirs(
        "result/" + dataname,
        exist_ok=True,
    )

    start = time.perf_counter()

    print("\n" + "=" * 78)
    print("FINAL SP-LSNTA-GeDGC (BIMODAL)")
    print("Dataset:", base_dataset)
    print("Checkpoint/result name:", dataname)
    print("Module 1: SP=True, private_beta=0.10")
    print(
        "Module 2: LSNTA=True "
        "(Latent SpectralNet + Feature-Gated MNN + UOT)"
    )
    print("Latent SpectralNet=True")
    print(
        "Feature-Gated MNN=True: "
        "lambda=0.20, K=15, power=1.00, min_gate=0.20"
    )
    print(
        "UOT=True: lambda=0.15, K=3, "
        "UOT_FeatureGate=False, MassGate=False"
    )
    print("Leiden=False")
    print("Residual/Causal/Agent/Reward=False")
    print("=" * 78)

    # --------------------------------------------------------
    # 1. Fresh SP pretraining
    # --------------------------------------------------------
    pretrained = ensure_pretrained(pretrain_opt, raw_data, label, c, dataname, info)
    setup_seed(seed)

    # --------------------------------------------------------
    # 2. Latent SpectralNet
    # --------------------------------------------------------
    print(
        "Use Latent SpectralNet: "
        "z_shared -> SpectralNet"
    )

    z_shared = get_pretrained_shared_latent(
        raw_data,
        dataname,
    )

    spectral_net = SpectralNet(
        n_clusters=c,
        should_use_ae=True,
        should_use_siamese=True,
        ae_hiddens=[512, c],
        siamese_hiddens=[256, c],
        spectral_hiddens=[256, c],
    )

    spectral_net.fit(z_shared)

    spectral_dist = spectral_net.predict(
        z_shared
    )

    if not torch.is_tensor(
        spectral_dist
    ):
        spectral_dist = torch.as_tensor(
            spectral_dist
        )

    spectral_dist = (
        spectral_dist
        .detach()
        .cpu()
        .float()
    )

    torch.save(
        spectral_dist,
        "Intermediate_data/"
        + dataname
        + "/LatentSpectralNet_D.pkl",
    )

    # --------------------------------------------------------
    # 3. Original GeDGC graph construction
    # --------------------------------------------------------
    neighbors = Form_data.cal_neighbors_D(
        spectral_dist,
        GRAPH_K,
    )

    del spectral_dist
    del z_shared
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = Form_data.form_data(
        raw_data,
        label,
        neighbors,
    )

    if isinstance(raw_data, list):
        raw_data.clear()

    del neighbors
    gc.collect()

    # --------------------------------------------------------
    # 4. FINAL train.py:
    #    Feature-Gated MNN -> UOT -> GeDGC joint optimization
    # --------------------------------------------------------
    setup_seed(seed)
    acc, nmi, ari, pur, fmi = train_opt(
        data,
        label,
        dataname,
        seed=seed,
    )

    run_time = time.perf_counter() - start
    score = (nmi + ari + fmi) / 3.0

    result = {
        "Dataset": base_dataset,
        "ACC": float(acc),
        "NMI": float(nmi),
        "ARI": float(ari),
        "Purity": float(pur),
        "FMI": float(fmi),
        "Score": float(score),
        "Time": float(run_time),
    }

    print(
        "| FINAL ACC = {:.6f} NMI = {:.6f} "
        "ARI = {:.6f} Purity = {:.6f} "
        "FMI = {:.6f} Score = {:.6f} "
        "TIME = {:.6f}".format(
            acc,
            nmi,
            ari,
            pur,
            fmi,
            score,
            run_time,
        )
    )

    return finish_run('SP-LSNTA-GeDGC', base_dataset, seed,
                      (acc, nmi, ari, pur), run_time, info, pretrained)

if __name__ == "__main__":
    cli('SP-LSNTA-GeDGC', run_one_dataset)
