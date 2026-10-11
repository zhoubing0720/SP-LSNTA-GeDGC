# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser(description="Historical Kidney construction with corrected feature header parsing")
parser.add_argument("--raw-dir", required=True, help="Directory containing the six uncompressed GEO files")
parser.add_argument("--output", default=str(Path(__file__).resolve().with_name("Kidney_4753_annotations_v1.h5mu")))
args = parser.parse_args()
output_path = Path(args.output).resolve()
if output_path.exists():
    raise FileExistsError("Refusing to overwrite an existing output: " + str(output_path))
output_path.parent.mkdir(parents=True, exist_ok=True)
os.chdir(Path(args.raw_dir).resolve())

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import anndata as ad

try:
    from mudata import MuData
except ImportError:
    import muon as mu
    MuData = mu.MuData


TARGET_CELLS = 4753
TARGET_RNA_FEATURES = 5000
TARGET_ATAC_FEATURES = 10000
RANDOM_SEED = 0


def read_names(path):
    # CSV headers are parsed by pandas; matrix rows correspond to data rows only.
    frame = pd.read_csv(path, dtype=str)
    return frame.iloc[:, 0].astype(str).str.strip().tolist()


def top_sparse_variance_features(X, n_top):
    """
    X: cells × features sparse matrix
    返回方差最高的 n_top 个特征下标
    """
    X = X.tocsr()
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    var = mean_sq - mean ** 2
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)

    n_top = min(n_top, X.shape[1])
    idx = np.argsort(var)[-n_top:]
    idx = idx[np.argsort(var[idx])[::-1]]
    return idx


def stratified_sample(obs, target_n, label_col="cell_type", seed=0):
    """
    按细胞类别比例抽样，尽量保留 14 类结构。
    """
    rng = np.random.default_rng(seed)
    selected = []

    counts = obs[label_col].value_counts()
    total = len(obs)

    # 每类按比例分配数量，至少保留 1 个
    alloc = {}
    for cls, cnt in counts.items():
        n_cls = int(round(cnt / total * target_n))
        n_cls = max(1, min(n_cls, cnt))
        alloc[cls] = n_cls

    # 修正总数到 target_n
    current = sum(alloc.values())

    while current > target_n:
        # 从分配数最多且大于 1 的类别里减
        candidates = [k for k, v in alloc.items() if v > 1]
        k = max(candidates, key=lambda x: alloc[x])
        alloc[k] -= 1
        current -= 1

    while current < target_n:
        # 从还有剩余细胞的类别里加
        candidates = [k for k, v in alloc.items() if v < counts[k]]
        if not candidates:
            break
        k = max(candidates, key=lambda x: counts[x] - alloc[x])
        alloc[k] += 1
        current += 1

    for cls, n_cls in alloc.items():
        idx = obs.index[obs[label_col] == cls].to_numpy()
        chosen = rng.choice(idx, size=n_cls, replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected)
    rng.shuffle(selected)
    return selected.tolist()


print("1. 读取 cell 信息...")

rna_cell = pd.read_csv("GSM3271044_RNA_mouse_kidney_cell.txt")
atac_cell = pd.read_csv("GSM3271045_ATAC_mouse_kidney_cell.txt")

rna_cell["sample"] = rna_cell["sample"].astype(str).str.strip()
atac_cell["sample"] = atac_cell["sample"].astype(str).str.strip()

# 只保留有标签的 RNA 细胞
rna_cell = rna_cell.dropna(subset=["cell_name"]).copy()
rna_cell["cell_name"] = rna_cell["cell_name"].astype(str).str.strip()

# RNA / ATAC 配对细胞
paired = pd.merge(
    rna_cell,
    atac_cell,
    on="sample",
    how="inner",
    suffixes=("_rna", "_atac")
)

paired = paired.drop_duplicates(subset=["sample"]).copy()
paired = paired.set_index("sample")

print("配对且有标签细胞数:", paired.shape[0])
print("类别数:", paired["cell_name"].nunique())
print(paired["cell_name"].value_counts())

print("\n2. 读取 gene / peak 名称...")

rna_genes = read_names("GSM3271044_RNA_mouse_kidney_gene.txt")
atac_peaks = read_names("GSM3271045_ATAC_mouse_kidney_peak.txt")

print("RNA gene names:", len(rna_genes))
print("ATAC peak names:", len(atac_peaks))

print("\n3. 读取 MatrixMarket count 矩阵，这一步可能需要几分钟...")

rna_mtx = mmread("GSM3271044_RNA_mouse_kidney_gene_count.txt").tocsr()
atac_mtx = mmread("GSM3271045_ATAC_mouse_kidney_peak_count.txt").tocsr()

assert len(rna_genes) == rna_mtx.shape[0], "RNA names/count rows differ"
assert len(atac_peaks) == atac_mtx.shape[0], "ATAC names/count rows differ"
print("RNA raw matrix:", rna_mtx.shape)
print("ATAC raw matrix:", atac_mtx.shape)

# 矩阵是 features × cells，转成 cells × features
rna_X_all = rna_mtx.T.tocsr()
atac_X_all = atac_mtx.T.tocsr()

# cell 文件顺序就是矩阵列顺序
rna_all_cells = pd.read_csv("GSM3271044_RNA_mouse_kidney_cell.txt")["sample"].astype(str).str.strip().tolist()
atac_all_cells = pd.read_csv("GSM3271045_ATAC_mouse_kidney_cell.txt")["sample"].astype(str).str.strip().tolist()

rna_index = {c: i for i, c in enumerate(rna_all_cells)}
atac_index = {c: i for i, c in enumerate(atac_all_cells)}

paired_cells = paired.index.tolist()

rna_rows = [rna_index[c] for c in paired_cells]
atac_rows = [atac_index[c] for c in paired_cells]

rna_X = rna_X_all[rna_rows, :]
atac_X = atac_X_all[atac_rows, :]

obs = pd.DataFrame(index=paired_cells)
obs["cell_type"] = pd.Categorical(paired.loc[paired_cells, "cell_name"].values)

print("\n配对矩阵:")
print("RNA paired:", rna_X.shape)
print("ATAC paired:", atac_X.shape)
print("Classes:", obs["cell_type"].nunique())

print("\n4. 删除 RNA 或 ATAC 全 0 细胞...")

rna_sum = np.asarray(rna_X.sum(axis=1)).ravel()
atac_sum = np.asarray(atac_X.sum(axis=1)).ravel()
keep_nonzero = (rna_sum > 0) & (atac_sum > 0)

rna_X = rna_X[keep_nonzero, :]
atac_X = atac_X[keep_nonzero, :]
obs = obs.iloc[np.where(keep_nonzero)[0]].copy()

print("非零细胞数:", obs.shape[0])
print("Classes:", obs["cell_type"].nunique())

print("\n5. 抽样到 4753 个细胞，尽量对齐 GeDGC Table I...")

if obs.shape[0] > TARGET_CELLS:
    selected_cells = stratified_sample(obs, TARGET_CELLS, label_col="cell_type", seed=RANDOM_SEED)
    selected_pos = obs.index.get_indexer(selected_cells)
    rna_X = rna_X[selected_pos, :]
    atac_X = atac_X[selected_pos, :]
    obs = obs.loc[selected_cells].copy()
else:
    print("警告：细胞数不足 4753，将保留全部细胞。")

print("最终细胞数:", obs.shape[0])
print("最终类别数:", obs["cell_type"].nunique())
print(obs["cell_type"].value_counts())

print("\n6. 选择高变 RNA genes 和 ATAC peaks...")

rna_feature_idx = top_sparse_variance_features(rna_X, TARGET_RNA_FEATURES)
atac_feature_idx = top_sparse_variance_features(atac_X, TARGET_ATAC_FEATURES)

rna_X = rna_X[:, rna_feature_idx].tocsr()
atac_X = atac_X[:, atac_feature_idx].tocsr()

rna_var = pd.DataFrame(index=np.array(rna_genes)[rna_feature_idx].astype(str))
atac_var = pd.DataFrame(index=np.array(atac_peaks)[atac_feature_idx].astype(str))

print("RNA final:", rna_X.shape)
print("ATAC final:", atac_X.shape)

print("\n7. 构建 h5mu 文件...")

adata_rna = ad.AnnData(
    X=rna_X.astype(np.float32),
    obs=obs.copy(),
    var=rna_var
)

adata_atac = ad.AnnData(
    X=atac_X.astype(np.float32),
    obs=obs.copy(),
    var=atac_var
)

adata_rna.var_names_make_unique()
adata_atac.var_names_make_unique()

mdata = MuData({
    "rna": adata_rna,
    "atac": adata_atac
})

mdata.write_h5mu(str(output_path))

print("\n转换完成：", output_path)
print("RNA shape:", mdata["rna"].shape)
print("ATAC shape:", mdata["atac"].shape)
print("Classes:", mdata["rna"].obs["cell_type"].nunique())
print(mdata["rna"].obs["cell_type"].value_counts())