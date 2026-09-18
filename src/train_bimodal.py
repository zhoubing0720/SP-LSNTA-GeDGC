import os
import gc
import tempfile
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
)
from tools.utils import cluster_acc, purity, to_numpy
from tools.early_stopping import EarlyStopping
from tqdm import trange
from tools.load_data import Cell
from tools.runtime_paths import checkpoint_file, result_file


def gen_x(mean, std, J):
    x_samples = []
    v_size = mean.size()
    for _ in range(J):
        x_samples.append(
            mean + torch.mul(std, torch.randn(v_size, device=mean.device))
        )
    return x_samples


def compute_weight(inputs, similarity_type="Gauss"):
    dist = 0
    for input_ in inputs:
        dist += torch.sum(
            torch.pow(
                input_ - input_[:, :, 0].unsqueeze(2),
                2,
            ),
            dim=1,
        )
    dist = F.normalize(dist, dim=1)

    if similarity_type == "Gauss":
        gauss = torch.exp(-dist)
        if gauss.shape[1] == 1:
            gauss[:, 0] = 1
        else:
            gauss[:, 0] = torch.sum(
                gauss[:, 1:],
                dim=1,
            )
        simi = torch.div(
            gauss,
            torch.sum(gauss, dim=1, keepdim=True),
        )
    else:
        n = inputs[0].size(-1)
        simi = torch.ones(1, n) / (n - 1)
        simi[0, 0] = 1
        simi = torch.mul(
            torch.ones(inputs[0].size(0), 1),
            simi,
        )
        simi = torch.div(
            simi,
            torch.sum(simi, dim=1, keepdim=True),
        )

    return simi


# ============================================================================
# FINAL BIMODAL SP-LSNTA-GeDGC
#
# Retained from M2:
#   SP private_beta=0.10
#   Latent SpectralNet (implemented in run.py)
#   MNN lambda=0.20, K=15
#   UOT lambda=0.15, K=3
#
# Only new change:
#   MNN Feature Gate power=1.00, min_gate=0.20
#
# Explicitly OFF:
#   UOT feature gate, Mass Gate, Leiden, residual classifier,
#   cross-modal residual, VQ/prototype, uncertainty, causal, agent, reward.
# ============================================================================


def _get_center_inputs(data):
    centers = []
    valid_indices = []
    for idx, view_data in enumerate(data):
        if (
            torch.is_tensor(view_data)
            and view_data.dim() == 3
            and view_data.size(2) >= 1
        ):
            centers.append(view_data[:, :, 0])
            valid_indices.append(idx)
    return centers, valid_indices


def _compute_view_latents_from_pretrained_vae(
    vae,
    data,
    device,
    batch_size=1024,
):
    centers, valid_indices = _get_center_inputs(data)
    if len(centers) == 0:
        return None, valid_indices

    vae = vae.to(device)
    vae.eval()

    n_cells = centers[0].shape[0]
    view_latents = [[] for _ in centers]

    with torch.no_grad():
        for start in range(0, n_cells, int(batch_size)):
            end = min(start + int(batch_size), n_cells)
            inputs = [
                x[start:end].to(device).float()
                for x in centers
            ]

            if hasattr(vae, "get_view_latent"):
                means, _ = vae.get_view_latent(inputs)
            else:
                means = []
                for view in range(len(inputs)):
                    mean, _ = vae.vaes[view].get_latent(
                        inputs[view]
                    )
                    means.append(mean)

            for view, mean in enumerate(means):
                view_latents[view].append(
                    mean.detach().cpu()
                )

    view_latents = [
        torch.cat(chunks, dim=0).float()
        for chunks in view_latents
    ]

    vae = vae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return view_latents, valid_indices


def _chunked_cosine_knn(
    x,
    k=15,
    chunk_size=512,
):
    x = F.normalize(
        x.detach().cpu().float(),
        p=2,
        dim=1,
    )
    n = int(x.shape[0])
    k = int(min(max(1, int(k)), n - 1))

    all_idx = []

    with torch.no_grad():
        for start in range(
            0,
            n,
            int(chunk_size),
        ):
            end = min(
                start + int(chunk_size),
                n,
            )

            sim = torch.mm(
                x[start:end],
                x.t(),
            )

            local_row = torch.arange(end - start)
            global_col = torch.arange(start, end)
            sim[local_row, global_col] = -1e9

            idx = torch.topk(
                sim,
                k=k,
                dim=1,
                largest=True,
            ).indices

            all_idx.append(idx.cpu())

    return (
        torch.cat(all_idx, dim=0)
        .numpy()
        .astype(np.int64, copy=False)
    )


def _build_mnn_reliable_neighbor_matrix(
    knn_indices_by_view,
    min_keep=1,
):
    n = int(knn_indices_by_view[0].shape[0])
    k = int(knn_indices_by_view[0].shape[1])
    n_views = len(knn_indices_by_view)

    knn_sets = []
    rank_maps = []

    for knn in knn_indices_by_view:
        sets = [
            set(int(v) for v in knn[i].tolist())
            for i in range(n)
        ]
        knn_sets.append(sets)

        ranks = []
        for i in range(n):
            ranks.append(
                {
                    int(j): rank
                    for rank, j
                    in enumerate(knn[i].tolist())
                }
            )
        rank_maps.append(ranks)

    selected_lists = []
    count_mutual = 0
    count_intersection = 0
    count_fallback = 0
    total_selected = 0

    for i in range(n):
        intersection = set.intersection(
            *[
                knn_sets[v][i]
                for v in range(n_views)
            ]
        )

        mutual = []
        for j in intersection:
            if all(
                i in knn_sets[v][int(j)]
                for v in range(n_views)
            ):
                mutual.append(int(j))

        if len(mutual) >= int(min_keep):
            selected = mutual
            count_mutual += 1

        elif len(intersection) > 0:
            selected = [
                int(j)
                for j in intersection
            ]
            count_intersection += 1

        else:
            union = set.union(
                *[
                    knn_sets[v][i]
                    for v in range(n_views)
                ]
            )

            scored = []
            for j in union:
                score = np.mean(
                    [
                        rank_maps[v][i].get(
                            int(j),
                            k,
                        )
                        for v in range(n_views)
                    ]
                )
                scored.append(
                    (float(score), int(j))
                )

            scored.sort(
                key=lambda item: item[0]
            )

            selected = [
                j
                for _, j in scored[
                    :max(1, int(min_keep))
                ]
            ]
            count_fallback += 1

        selected = [
            int(j)
            for j in selected
            if int(j) != i
        ][:k]

        if len(selected) == 0:
            selected = [
                int(
                    knn_indices_by_view[0][i, 0]
                )
            ]

        selected_lists.append(selected)
        total_selected += len(selected)

    max_len = max(
        len(values)
        for values in selected_lists
    )

    neighbor_ids = np.zeros(
        (n, max_len),
        dtype=np.int64,
    )
    neighbor_mask = np.zeros(
        (n, max_len),
        dtype=np.float32,
    )

    for i, selected in enumerate(selected_lists):
        length = len(selected)
        neighbor_ids[
            i,
            :length,
        ] = np.asarray(
            selected,
            dtype=np.int64,
        )
        neighbor_mask[
            i,
            :length,
        ] = 1.0

    print(
        "MNN stats: k={}, views={}, mean_selected={:.4f}, "
        "mutual_ratio={:.6f}, intersection_ratio={:.6f}, "
        "fallback_ratio={:.6f}".format(
            k,
            n_views,
            total_selected / float(max(n, 1)),
            count_mutual / float(max(n, 1)),
            count_intersection / float(max(n, 1)),
            count_fallback / float(max(n, 1)),
        )
    )

    return neighbor_ids, neighbor_mask


def _compute_feature_gate(
    x_center,
    feature_gate_power=1.0,
    feature_min_gate=0.20,
    row_chunk_size=256,
):
    """
    Feature reliability gate from the retained research implementation.

    High-variance and sufficiently non-zero features receive stronger
    neighbor aggregation. Low-quality/sparse features remain closer to
    the original center cell.

    power=0 exactly degenerates to ordinary MNN.
    """
    if feature_gate_power <= 0:
        return torch.ones(
            x_center.shape[1],
            dtype=x_center.dtype,
            device=x_center.device,
        )

    n_cells = int(x_center.shape[0])
    n_features = int(x_center.shape[1])

    sum_x = torch.zeros(
        n_features,
        dtype=torch.float64,
        device=x_center.device,
    )
    sum_x2 = torch.zeros(
        n_features,
        dtype=torch.float64,
        device=x_center.device,
    )
    nonzero_count = torch.zeros(
        n_features,
        dtype=torch.float64,
        device=x_center.device,
    )

    with torch.no_grad():
        for start in range(
            0,
            n_cells,
            int(row_chunk_size),
        ):
            end = min(
                start + int(row_chunk_size),
                n_cells,
            )
            chunk = x_center[start:end]
            work = chunk.to(dtype=torch.float64)

            sum_x.add_(
                work.sum(dim=0)
            )
            sum_x2.add_(
                (work * work).sum(dim=0)
            )
            nonzero_count.add_(
                (chunk.abs() > 1e-12).sum(
                    dim=0,
                    dtype=torch.float64,
                )
            )

        count = float(n_cells)
        mean = sum_x / count
        var = (
            sum_x2 / count
            - mean.square()
        ).clamp_min_(0.0)

        nz = (
            nonzero_count / count
        ).clamp_(0.0, 1.0)

        raw = (
            torch.log1p(var)
            * torch.sqrt(
                torch.clamp(
                    nz,
                    min=1e-6,
                )
            )
        )

        raw_min = torch.min(raw)
        raw_max = torch.max(raw)

        gate = (
            raw - raw_min
        ) / (
            raw_max - raw_min + 1e-12
        )

        gate = torch.pow(
            torch.clamp(
                gate,
                min=0.0,
                max=1.0,
            ),
            float(feature_gate_power),
        )

        gate = (
            float(feature_min_gate)
            + (
                1.0 - float(feature_min_gate)
            ) * gate
        )

    return gate.to(
        dtype=x_center.dtype,
        device=x_center.device,
    )


def aggregate_feature_gated_mnn(
    data,
    vae,
    lambda_input=0.20,
    mnn_top_k=15,
    min_keep=1,
    feature_gate_power=1.0,
    feature_min_gate=0.20,
    latent_batch_size=1024,
    knn_chunk_size=512,
    feature_chunk_size=64,
    device=None,
):
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    view_latents, valid_indices = (
        _compute_view_latents_from_pretrained_vae(
            vae,
            data,
            device=device,
            batch_size=latent_batch_size,
        )
    )

    knn_by_view = [
        _chunked_cosine_knn(
            latent,
            k=mnn_top_k,
            chunk_size=knn_chunk_size,
        )
        for latent in view_latents
    ]

    ids_np, mask_np = (
        _build_mnn_reliable_neighbor_matrix(
            knn_by_view,
            min_keep=min_keep,
        )
    )

    neighbor_ids = (
        torch.from_numpy(ids_np).long()
    )
    neighbor_mask = (
        torch.from_numpy(mask_np).float()
    )
    count = (
        neighbor_mask
        .sum(dim=1, keepdim=True)
        .clamp_min(1.0)
    )

    out_data = list(data)

    for local_view, data_index in enumerate(
        valid_indices
    ):
        view_data = out_data[data_index]
        x_center_all = view_data[:, :, 0]

        gate = _compute_feature_gate(
            x_center_all,
            feature_gate_power=feature_gate_power,
            feature_min_gate=feature_min_gate,
        )

        print(
            "Feature gate view {}: power={:.2f}, "
            "min={:.6f}, max={:.6f}, mean={:.6f}".format(
                local_view,
                feature_gate_power,
                float(gate.min()),
                float(gate.max()),
                float(gate.mean()),
            )
        )

        n_cells = int(x_center_all.shape[0])
        n_features = int(x_center_all.shape[1])

        np_dtype = (
            np.float32
            if x_center_all.element_size() <= 4
            else np.float64
        )

        temp_handle = tempfile.NamedTemporaryFile(
            prefix="gedgc_m3_gate_",
            suffix=".mmap",
            delete=False,
        )
        temp_path = temp_handle.name
        temp_handle.close()

        mmap_array = None

        try:
            print(
                "Feature-gated MNN view {} low-memory aggregation: "
                "{} cells x {} features".format(
                    local_view,
                    n_cells,
                    n_features,
                )
            )

            mmap_array = np.memmap(
                temp_path,
                mode="w+",
                dtype=np_dtype,
                shape=(
                    n_cells,
                    n_features,
                ),
            )

            with torch.no_grad():
                for start in range(
                    0,
                    n_cells,
                    int(feature_chunk_size),
                ):
                    end = min(
                        start + int(feature_chunk_size),
                        n_cells,
                    )

                    ids = neighbor_ids[
                        start:end
                    ]
                    mask = neighbor_mask[
                        start:end
                    ]
                    denom = count[
                        start:end
                    ]

                    neigh = x_center_all[ids]

                    neigh_mean = (
                        (
                            neigh
                            * mask.unsqueeze(-1)
                            .type_as(neigh)
                        ).sum(dim=1)
                        / denom.type_as(neigh)
                    )

                    center = x_center_all[
                        start:end
                    ]

                    result = (
                        center
                        + float(lambda_input)
                        * (
                            neigh_mean - center
                        )
                        * gate.unsqueeze(0)
                    )

                    mmap_array[
                        start:end
                    ] = (
                        result
                        .detach()
                        .cpu()
                        .numpy()
                    )

            mmap_array.flush()

            with torch.no_grad():
                for start in range(
                    0,
                    n_cells,
                    256,
                ):
                    end = min(
                        start + 256,
                        n_cells,
                    )

                    chunk_np = np.array(
                        mmap_array[
                            start:end
                        ],
                        copy=True,
                    )

                    view_data[
                        start:end,
                        :,
                        0,
                    ].copy_(
                        torch.from_numpy(
                            chunk_np
                        ).to(
                            dtype=view_data.dtype,
                            device=view_data.device,
                        )
                    )

            out_data[data_index] = view_data

        finally:
            if mmap_array is not None:
                mmap_array.flush()
                del mmap_array

            gc.collect()

            try:
                os.remove(temp_path)
            except OSError:
                pass

    print(
        "Feature-gated MNN complete: "
        "lambda={:.2f}, top_k={}, power={:.2f}, min_gate={:.2f}".format(
            lambda_input,
            mnn_top_k,
            feature_gate_power,
            feature_min_gate,
        )
    )

    return out_data


def _unbalanced_sinkhorn(
    cost,
    entropic_reg=0.05,
    mass_reg=1.0,
    max_iter=60,
    tol=1e-6,
):
    n_source, n_target = cost.shape
    eps = float(entropic_reg)
    tau = float(mass_reg)

    dtype = cost.dtype
    device = cost.device
    tiny = torch.finfo(dtype).eps

    a = torch.full(
        (n_source,),
        1.0 / float(n_source),
        dtype=dtype,
        device=device,
    )
    b = torch.full(
        (n_target,),
        1.0 / float(n_target),
        dtype=dtype,
        device=device,
    )

    kernel = torch.exp(
        -torch.clamp(
            cost,
            min=0.0,
            max=50.0,
        ) / eps
    ).clamp_min(tiny)

    exponent = tau / (tau + eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(int(max_iter)):
        previous_u = u.clone()

        kv = torch.matmul(
            kernel,
            v,
        ).clamp_min(tiny)

        u = torch.pow(
            a / kv,
            exponent,
        )

        ktu = torch.matmul(
            kernel.transpose(0, 1),
            u,
        ).clamp_min(tiny)

        v = torch.pow(
            b / ktu,
            exponent,
        )

        if (
            torch.max(
                torch.abs(
                    u - previous_u
                )
            ).item()
            < float(tol)
        ):
            break

    return (
        u.unsqueeze(1)
        * kernel
        * v.unsqueeze(0)
    )


def _topk_row_normalize(
    weight,
    top_k=3,
):
    rows, cols = weight.shape
    k = int(
        min(
            max(1, int(top_k)),
            cols,
        )
    )

    values, indices = torch.topk(
        weight,
        k=k,
        dim=1,
        largest=True,
    )

    kept = torch.zeros_like(weight)
    kept.scatter_(
        1,
        indices,
        values,
    )

    return (
        kept
        / kept.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
    )


def aggregate_uot_no_gate(
    data,
    vae,
    lambda_uot=0.15,
    uot_batch_size=256,
    uot_top_k=3,
    entropic_reg=0.05,
    mass_reg=1.0,
    sinkhorn_iter=60,
    exclude_paired_diagonal=True,
    seed=0,
    latent_batch_size=1024,
    device=None,
):
    """
    Same pure-geometric UOT path as M2.
    Feature Gate and Mass Gate remain OFF.
    """
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    centers, valid_indices = _get_center_inputs(data)

    if len(centers) != 2:
        raise ValueError(
            "M3 UOT requires exactly two modalities."
        )

    view_latents, _ = (
        _compute_view_latents_from_pretrained_vae(
            vae,
            data,
            device=device,
            batch_size=latent_batch_size,
        )
    )

    z0_all = F.normalize(
        view_latents[0].float(),
        p=2,
        dim=1,
    )
    z1_all = F.normalize(
        view_latents[1].float(),
        p=2,
        dim=1,
    )

    x0_all = centers[0]
    x1_all = centers[1]
    n_cells = int(x0_all.shape[0])

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        int(seed) + 5151
    )

    permutation = torch.randperm(
        n_cells,
        generator=generator,
    )

    shift0_sum = 0.0
    shift1_sum = 0.0
    total_cells = 0
    num_batches = 0

    with torch.no_grad():
        for batch_start in range(
            0,
            n_cells,
            int(uot_batch_size),
        ):
            indices = permutation[
                batch_start:
                batch_start + int(
                    uot_batch_size
                )
            ]

            current_size = int(
                indices.numel()
            )

            if current_size < 2:
                continue

            z0 = z0_all[
                indices
            ].to(device)

            z1 = z1_all[
                indices
            ].to(device)

            cost = torch.cdist(
                z0,
                z1,
                p=2,
            ).pow(2)

            cost_min = cost.min()
            cost_max = cost.max()

            cost = (
                cost - cost_min
            ) / (
                cost_max
                - cost_min
                + 1e-12
            )

            if exclude_paired_diagonal:
                diag = torch.arange(
                    current_size,
                    device=device,
                )
                cost[
                    diag,
                    diag,
                ] = 1.5

            coupling = _unbalanced_sinkhorn(
                cost,
                entropic_reg=entropic_reg,
                mass_reg=mass_reg,
                max_iter=sinkhorn_iter,
            )

            row_weights = (
                _topk_row_normalize(
                    coupling,
                    top_k=uot_top_k,
                )
            )

            column_weights = (
                _topk_row_normalize(
                    coupling.transpose(0, 1),
                    top_k=uot_top_k,
                )
            )

            x0 = x0_all[
                indices
            ].to(device).float()

            x1 = x1_all[
                indices
            ].to(device).float()

            neighbor0 = torch.matmul(
                row_weights,
                x0,
            )
            neighbor1 = torch.matmul(
                column_weights,
                x1,
            )

            shift0 = (
                float(lambda_uot)
                * (
                    neighbor0 - x0
                )
            )

            shift1 = (
                float(lambda_uot)
                * (
                    neighbor1 - x1
                )
            )

            new0 = x0 + shift0
            new1 = x1 + shift1

            data[
                valid_indices[0]
            ][
                indices,
                :,
                0,
            ] = new0.detach().cpu().to(
                data[
                    valid_indices[0]
                ].dtype
            )

            data[
                valid_indices[1]
            ][
                indices,
                :,
                0,
            ] = new1.detach().cpu().to(
                data[
                    valid_indices[1]
                ].dtype
            )

            shift0_sum += float(
                torch.norm(
                    shift0,
                    p=2,
                    dim=1,
                ).sum().cpu()
            )
            shift1_sum += float(
                torch.norm(
                    shift1,
                    p=2,
                    dim=1,
                ).sum().cpu()
            )

            total_cells += current_size
            num_batches += 1

    print(
        "UOT aggregation complete: lambda={:.2f}, "
        "top_k={}, batches={}, "
        "mean_shift_view0={:.6f}, "
        "mean_shift_view1={:.6f}, "
        "feature_gate=False, mass_gate=False".format(
            lambda_uot,
            uot_top_k,
            num_batches,
            shift0_sum / float(
                max(total_cells, 1)
            ),
            shift1_sum / float(
                max(total_cells, 1)
            ),
        )
    )

    return data


def train_model(
    vae,
    classifier,
    GMM,
    optimizer,
    lr_scheduler,
    dataloader,
    dataname,
    epoch_num,
    device,
):
    vae = vae.to(device)
    classifier = classifier.to(device)
    GMM = GMM.to(device)

    avg_loss = []
    early_stopping = EarlyStopping(
        patience=10,
        delta=1e-3,
    )

    t = trange(
        epoch_num,
        leave=True,
    )

    for epoch in t:
        total_loss = []
        label = []
        label_pred = []
        recon_loss = []

        for inputs, labels in dataloader:
            n_samples = inputs[0].size(0)
            n_n = inputs[0].size(2)

            inputs = [
                input_.to(device)
                for input_ in inputs
            ]

            all_inputs = []
            all_targets = []

            for data_v in inputs:
                temp_data1 = []
                temp_data2 = []

                for idx in range(n_n):
                    temp_data1.append(
                        data_v[:, :, idx]
                    )
                    temp_data2.append(
                        data_v[:, :, 0]
                    )

                all_inputs.append(
                    torch.cat(
                        temp_data1,
                        dim=0,
                    )
                )
                all_targets.append(
                    torch.cat(
                        temp_data2,
                        dim=0,
                    )
                )

            label.append(labels)

            weight = compute_weight(inputs)

            weight_temp = [
                weight[:, idx]
                for idx in range(n_n)
            ]
            weight = torch.cat(weight_temp)

            vae.eval()
            classifier.eval()

            with torch.no_grad():
                x_mean, _ = vae.get_latent(
                    all_inputs
                )
                pc = classifier(
                    x_mean
                ).data

            pc = torch.mul(
                pc,
                weight.view(-1, 1),
            )

            pc_temp = 0
            for idx in range(n_n):
                pc_temp = (
                    pc_temp
                    + pc[
                        idx * n_samples:
                        (idx + 1) * n_samples,
                        :
                    ]
                )

            pc = torch.cat(
                [
                    pc_temp
                    for _ in range(n_n)
                ],
                dim=0,
            )

            vae.train()
            classifier.train()
            GMM.train()

            j = 1
            x_mean, x_logvar = (
                vae.get_latent(
                    all_inputs
                )
            )

            x_samples = gen_x(
                x_mean,
                torch.exp(
                    0.5 * x_logvar
                ),
                j,
            )

            elbo = 0

            for idx in range(j):
                x_re = vae.get_recon(
                    x_samples[idx]
                )

                recon = 0

                for view in range(
                    len(x_re)
                ):
                    recon += (
                        -0.5
                        * torch.sum(
                            torch.pow(
                                all_targets[view]
                                - x_re[view],
                                2,
                            ),
                            dim=1,
                        )
                    )

                elbo = (
                    elbo
                    + 0.1
                    * torch.sum(
                        torch.mul(
                            recon,
                            weight,
                        )
                    )
                )

            elbo = elbo / j

            recon_loss.append(
                (-elbo / n_samples).item()
            )

            cond_prob = classifier(
                x_samples[0]
            )

            elbo = (
                elbo
                + torch.sum(
                    torch.mul(
                        torch.sum(
                            -torch.mul(
                                cond_prob,
                                torch.log(
                                    cond_prob
                                    + 1e-10
                                ),
                            ),
                            dim=-1,
                        ),
                        weight,
                    )
                )
            )

            elbo = (
                elbo
                + torch.sum(
                    torch.mul(
                        torch.sum(
                            torch.mul(
                                cond_prob,
                                torch.log(
                                    pc
                                    + 1e-10
                                ),
                            ),
                            dim=-1,
                        ),
                        weight,
                    )
                )
            )

            elbo = (
                elbo
                + GMM.log_prob(
                    x_mean,
                    x_logvar,
                    cond_prob,
                    weight,
                )
            )

            elbo = (
                elbo
                + torch.sum(
                    torch.mul(
                        0.5
                        * torch.sum(
                            x_logvar,
                            dim=-1,
                        ),
                        weight,
                    )
                )
            )

            loss = -elbo / n_samples

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(
                loss.item()
            )

            if np.isnan(loss.item()):
                return (
                    vae,
                    classifier,
                    GMM,
                    avg_loss,
                )

            vae.eval()
            classifier.eval()

            data_i = [
                data_v[:, :, 0].data
                for data_v in inputs
            ]

            with torch.no_grad():
                x_mean, _ = (
                    vae.get_latent(data_i)
                )
                pred_l = torch.max(
                    classifier(x_mean),
                    dim=-1,
                )

            label_pred.append(
                pred_l[-1]
            )

        lr_scheduler.step()

        label = torch.cat(
            label,
            dim=0,
        )
        label_pred = torch.cat(
            label_pred,
            dim=0,
        )

        nmi = normalized_mutual_info_score(
            to_numpy(label),
            to_numpy(label_pred),
        )
        ari = adjusted_rand_score(
            to_numpy(label),
            to_numpy(label_pred),
        )
        pur = purity(
            to_numpy(label),
            to_numpy(label_pred),
        )
        acc, _ = cluster_acc(
            to_numpy(label_pred),
            to_numpy(label),
        )

        avg_loss.append(
            np.mean(total_loss)
        )

        t.set_description(
            "|Epoch:{} Total loss={:3f} "
            "Reconstruction Loss={:6f} "
            "PUR={:5f} ARI={:5f} "
            "NMI={:5f} ACC={:6f}".format(
                epoch + 1,
                np.mean(total_loss),
                np.mean(recon_loss),
                pur,
                ari,
                nmi,
                acc,
            )
        )

        t.refresh()

        early_stopping(
            np.mean(total_loss)
        )

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return (
        vae,
        classifier,
        GMM,
        avg_loss,
    )


def train_opt(
    data,
    label,
    dataname,
    seed=0,
):
    if data[0].shape[0] > 1024:
        batch_size = 128
    else:
        batch_size = 16

    learning_rate = 0.0001
    weight_decay = 1e-6
    step_size = 50
    gama = 0.1
    epoch = 1000

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 78)
    print(
        "FINAL BIMODAL: SP + LSNTA "
        "+ Feature-Gated MNN + UOT"
    )
    print("=" * 78)
    print(
        "private_beta=0.10"
    )
    print(
        "MNN: lambda=0.20, K=15, "
        "feature_gate_power=1.00, min_gate=0.20"
    )
    print(
        "UOT: lambda=0.15, K=3, "
        "feature_gate=False, mass_gate=False"
    )
    print(
        "Disabled: Leiden / residual / "
        "cross-modal residual / VQ / uncertainty / "
        "causal / agent / reward"
    )
    print("=" * 78)

    print("Load pretrained SP model")

    vae = torch.load(checkpoint_file("pretrained_vae.pkl"))
    classifier = torch.load(checkpoint_file("pretrained_classifier.pkl"))
    GMM = torch.load(checkpoint_file("pretrained_GMM.pkl"))

    data_setting = (
        aggregate_feature_gated_mnn(
            data,
            vae=vae,
            lambda_input=0.20,
            mnn_top_k=15,
            min_keep=1,
            feature_gate_power=1.00,
            feature_min_gate=0.20,
            latent_batch_size=1024,
            knn_chunk_size=512,
            feature_chunk_size=64,
            device=device,
        )
    )

    data_setting = (
        aggregate_uot_no_gate(
            data_setting,
            vae=vae,
            lambda_uot=0.15,
            uot_batch_size=256,
            uot_top_k=3,
            entropic_reg=0.05,
            mass_reg=1.0,
            sinkhorn_iter=60,
            exclude_paired_diagonal=True,
            seed=seed,
            latent_batch_size=1024,
            device=device,
        )
    )

    # Reset formal optimization RNG.
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataloader = DataLoader(
        Cell(
            data_setting,
            label,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(
        list(vae.parameters())
        + list(classifier.parameters())
        + list(GMM.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    lr_scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gama,
    )

    vae, classifier, GMM, avg_loss = train_model(
        vae,
        classifier,
        GMM,
        optimizer,
        lr_scheduler,
        dataloader,
        dataname,
        epoch,
        device,
    )

    vae.eval()
    classifier.eval()
    GMM.eval()

    datai = [
        data_[:, :, 0]
        for data_ in data_setting
    ]

    dataloader = DataLoader(
        Cell(datai),
        batch_size=1024,
        shuffle=False,
    )

    x_mean = []
    x_logvar = []

    for inputs in dataloader:
        inputs = [
            input_.to(device)
            for input_ in inputs
        ]

        with torch.no_grad():
            x_m, x_v = (
                vae.get_latent(inputs)
            )

        x_mean.append(x_m)
        x_logvar.append(x_v)

    x_mean = torch.cat(
        x_mean,
        dim=0,
    )
    x_logvar = torch.cat(
        x_logvar,
        dim=0,
    )

    # === TRAJECTORY_LATENT_EXPORT ===
    import numpy as _traj_np

    _traj_path = result_file("Trajectory_latent_SP_LSNTA_GeDGC.npz")
    _traj_np.savez_compressed(
        _traj_path,
        latent=x_mean.detach().cpu().numpy(),
        y_true=_traj_np.asarray(label),
    )
    print("Saved trajectory latent ->", _traj_path)

    x_sample = gen_x(
        x_mean,
        torch.exp(
            0.5 * x_logvar
        ),
        1,
    )[0]

    with torch.no_grad():
        pred_label = torch.max(
            classifier(x_sample),
            dim=-1,
        )

    pred_np = to_numpy(
        pred_label[-1]
    )

    acc, _ = cluster_acc(
        pred_np,
        label,
    )

    nmi = normalized_mutual_info_score(
        label,
        pred_np,
    )

    ari = adjusted_rand_score(
        label,
        pred_np,
    )

    pur = purity(
        label,
        pred_np,
    )

    fmi = fowlkes_mallows_score(
        label,
        pred_np,
    )

    score = (
        nmi + ari + fmi
    ) / 3.0

    print(
        "| FINAL ACC = {:.6f} NMI = {:.6f} "
        "ARI = {:.6f} Purity = {:.6f} "
        "FMI = {:.6f} Score = {:.6f}".format(
            acc,
            nmi,
            ari,
            pur,
            fmi,
            score,
        )
    )

    return (
        acc,
        nmi,
        ari,
        pur,
        fmi,
    )
