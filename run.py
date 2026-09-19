#!/usr/bin/env python
"""Unified entry point for the seven paper datasets."""

import argparse
import csv
import gc
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CHECKPOINT_FILENAMES = (
    "pretrained_vae.pkl",
    "pretrained_GMM.pkl",
    "pretrained_classifier.pkl",
)
METRIC_FIELDS = ("ACC", "NMI", "ARI", "Purity", "FMI", "Score", "Time")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_experiment_config(config):
    """Check the public fixed configuration before a scientific run starts."""
    required = {
        "graph_k",
        "private_beta",
        "pretrain_vae_epochs",
        "pretrain_classifier_epochs",
        "training_epochs",
        "mnn_lambda",
        "mnn_k",
        "mnn_feature_gate_power",
        "mnn_min_gate",
        "uot_lambda",
        "uot_k",
        "uot_feature_gate",
        "uot_mass_gate",
        "spectral_ae_hiddens",
        "spectral_siamese_hiddens",
        "spectral_hiddens",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError("configs/experiment.json is missing keys: {}".format(missing))
    positive = ("graph_k", "pretrain_vae_epochs", "pretrain_classifier_epochs", "training_epochs", "mnn_k", "uot_k")
    for key in positive:
        if int(config[key]) < 1:
            raise ValueError("Experiment setting {!r} must be positive.".format(key))
    for key in ("private_beta", "mnn_lambda", "mnn_feature_gate_power", "mnn_min_gate", "uot_lambda"):
        if float(config[key]) < 0:
            raise ValueError("Experiment setting {!r} must be non-negative.".format(key))
    if bool(config["uot_feature_gate"]) or bool(config["uot_mass_gate"]):
        raise ValueError(
            "This release implements the manuscript configuration with both "
            "UOT gates disabled."
        )
    for key in ("spectral_ae_hiddens", "spectral_siamese_hiddens", "spectral_hiddens"):
        value = config[key]
        if not isinstance(value, list) or len(value) != 2 or value[1] != "n_clusters":
            raise ValueError(
                "Experiment setting {!r} must be [hidden_width, 'n_clusters'].".format(key)
            )
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one SP-LSNTA-GeDGC dataset per invocation."
    )
    parser.add_argument("--dataset", help="Canonical dataset name from configs/datasets.json")
    parser.add_argument("--seed", type=int, default=0, help="First/random seed (original default: 0)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit independent-run seeds")
    parser.add_argument("--repeats", type=int, default=1, help="Use consecutive seeds from --seed")
    parser.add_argument("--force-pretrain", action="store_true", help="Ignore reusable pretrained files")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and show the run plan")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--checkpoints-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args(argv)
    if not args.list_datasets and not args.dataset:
        parser.error("--dataset is required unless --list-datasets is used")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.seeds and args.repeats != 1:
        parser.error("use either --seeds or --repeats, not both")
    return args


def resolve_dataset(name, registry):
    normalized = name.lower().replace("_", "-")
    aliases = {
        "chen-2019": "Chen-2019",
        "chen-high": "Chen-2019",
        "kidney": "Kidney",
        "kidney-4753": "Kidney",
        "pbmc10k": "Pbmc10k",
        "opmultiome": "opmultiome",
        "multiome": "multiome",
        "multiome-high": "multiome",
        "tea-seq": "TEA-seq",
        "teaseq": "TEA-seq",
        "dyngen-branching": "dyngen_branching",
        "dyngen": "dyngen_branching",
    }
    canonical = aliases.get(normalized)
    if canonical not in registry:
        raise ValueError("Unknown dataset {!r}. Available: {}".format(name, list(registry)))
    return canonical, registry[canonical]


def setup_seed(seed):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def checkpoint_complete(path):
    path = Path(path)
    return all((path / filename).is_file() for filename in CHECKPOINT_FILENAMES)


def prepare_checkpoints(canonical_dir, search_dirs, force_pretrain):
    canonical_dir = Path(canonical_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if force_pretrain:
        return False, None
    if checkpoint_complete(canonical_dir):
        return True, canonical_dir
    for candidate in search_dirs:
        candidate = Path(candidate)
        if checkpoint_complete(candidate):
            for filename in CHECKPOINT_FILENAMES:
                shutil.copy2(candidate / filename, canonical_dir / filename)
            return True, candidate
    return False, None


def get_pretrained_shared_latent(raw_data, batch_size=1024):
    import torch
    from torch.utils.data import DataLoader
    from tools.load_data import Cell
    from tools.runtime_paths import checkpoint_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = torch.load(checkpoint_file("pretrained_vae.pkl"))
    vae = vae.to(device)
    vae.eval()
    loader = DataLoader(Cell(raw_data), batch_size=batch_size, shuffle=False)
    latent_chunks = []
    with torch.no_grad():
        for inputs in loader:
            inputs = [value.to(device).float() for value in inputs]
            mean, _ = vae.get_latent(inputs)
            latent_chunks.append(mean.detach().cpu())
    z_shared = torch.cat(latent_chunks, dim=0).float()
    print("Latent-SpectralNet z_shared shape:", tuple(z_shared.shape))
    vae = vae.to("cpu")
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return z_shared


def run_once(dataset_name, spec, experiment, seed, args):
    import torch
    from graph_embedding.spectralnet import SpectralNet
    from pretrain import pretrain_opt
    from tools import Form_data
    from tools.load_data import load_dataset
    from tools.runtime_paths import checkpoint_file, configure

    train_module = __import__(
        "train_bimodal" if spec["backend"] == "bimodal" else "train_nmodal"
    )
    setup_seed(seed)
    raw_data, label, n_clusters = load_dataset(spec, args.data_dir)

    checkpoint_dir = args.checkpoints_dir / dataset_name / ("seed_{}".format(seed))
    result_dir = args.results_dir / dataset_name / ("seed_{}".format(seed))
    legacy_name = spec["legacy_checkpoint_name"]
    search_dirs = []
    if seed == 0:
        search_dirs = [
            args.checkpoints_dir / dataset_name,
            args.checkpoints_dir / legacy_name,
            PROJECT_ROOT / "Intermediate_data" / legacy_name,
        ]
    reusable, source = prepare_checkpoints(
        checkpoint_dir, search_dirs, args.force_pretrain
    )
    configure(checkpoint_dir, result_dir)
    write_json(
        result_dir / "run_config.json",
        {
            "dataset": dataset_name,
            "seed": seed,
            "dataset_spec": spec,
            "experiment": experiment,
            "checkpoint_reused": bool(reusable),
            "checkpoint_source": str(source) if source else None,
        },
    )

    start = time.perf_counter()
    print("\n" + "=" * 78)
    print("SP-LSNTA-GeDGC | dataset={} | seed={}".format(dataset_name, seed))
    print("Backend:", spec["backend"])
    print("Checkpoint directory:", checkpoint_dir)
    print("Result directory:", result_dir)
    print("=" * 78)

    if reusable:
        print("Reuse complete pretrained checkpoint set from:", source)
    else:
        print("Complete pretrained checkpoint set not found; run pretraining.")
        pretrain_opt(raw_data, label, n_clusters, legacy_name, seed=seed, config=experiment)

    z_shared = get_pretrained_shared_latent(raw_data)
    spectral_net = SpectralNet(
        n_clusters=n_clusters,
        should_use_ae=True,
        should_use_siamese=True,
        ae_hiddens=[int(experiment["spectral_ae_hiddens"][0]), n_clusters],
        siamese_hiddens=[int(experiment["spectral_siamese_hiddens"][0]), n_clusters],
        spectral_hiddens=[int(experiment["spectral_hiddens"][0]), n_clusters],
    )
    spectral_net.fit(z_shared)
    spectral_dist = spectral_net.predict(z_shared)
    if not torch.is_tensor(spectral_dist):
        spectral_dist = torch.as_tensor(spectral_dist)
    spectral_dist = spectral_dist.detach().cpu().float()
    torch.save(spectral_dist, checkpoint_file("LatentSpectralNet_D.pkl"))

    neighbors = Form_data.cal_neighbors_D(spectral_dist, int(experiment["graph_k"]))
    del spectral_dist, z_shared
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    formed_data = Form_data.form_data(raw_data, label, neighbors)
    if isinstance(raw_data, list):
        raw_data.clear()
    del neighbors
    gc.collect()

    acc, nmi, ari, pur, fmi = train_module.train_opt(
        formed_data, label, legacy_name, seed=seed, config=experiment
    )
    elapsed = time.perf_counter() - start
    row = {
        "Dataset": dataset_name,
        "Seed": seed,
        "ACC": float(acc),
        "NMI": float(nmi),
        "ARI": float(ari),
        "Purity": float(pur),
        "FMI": float(fmi),
        "Score": float((nmi + ari + fmi) / 3.0),
        "Time": float(elapsed),
    }
    write_csv(result_dir / "metrics.csv", [row], list(row))
    return row


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(dataset_name, rows, results_dir):
    parent = Path(results_dir) / dataset_name
    run_fields = ["Dataset", "Seed"] + list(METRIC_FIELDS)
    write_csv(parent / "runs.csv", rows, run_fields)
    summary = {"Dataset": dataset_name, "Runs": len(rows)}
    for field in METRIC_FIELDS:
        values = [float(row[field]) for row in rows]
        summary[field + "_mean"] = sum(values) / float(len(values))
        summary[field + "_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    write_csv(parent / "summary.csv", [summary], list(summary))


def dry_run(dataset_name, spec, seeds, args):
    candidates = [spec["filename"]] + spec.get("legacy_filenames", [])
    existing = [args.data_dir / name for name in candidates if (args.data_dir / name).is_file()]
    print("Dataset:", dataset_name)
    print("Backend:", spec["backend"])
    print("Modalities:", [item["name"] for item in spec["modalities"]])
    print("Seeds:", seeds)
    print("Data candidates:", [str(args.data_dir / name) for name in candidates])
    print("Resolved data:", existing[0] if existing else "MISSING")
    for seed in seeds:
        path = args.checkpoints_dir / dataset_name / ("seed_{}".format(seed))
        print("seed {} pretrained complete: {}".format(seed, checkpoint_complete(path)))
    return 0 if existing else 2


def main(argv=None):
    args = parse_args(argv)
    registry = load_json(PROJECT_ROOT / "configs" / "datasets.json")
    if args.list_datasets:
        print("\n".join(registry))
        return 0
    dataset_name, spec = resolve_dataset(args.dataset, registry)
    experiment = validate_experiment_config(
        load_json(PROJECT_ROOT / "configs" / "experiment.json")
    )
    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + args.repeats))
    if args.dry_run:
        return dry_run(dataset_name, spec, seeds, args)
    rows = [run_once(dataset_name, spec, experiment, seed, args) for seed in seeds]
    write_summary(dataset_name, rows, args.results_dir)
    print("Completed {} run(s). Summary: {}".format(
        len(rows), args.results_dir / dataset_name / "summary.csv"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
