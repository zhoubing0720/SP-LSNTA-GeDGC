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
# M3 = M2 + Feature Gate ONLY
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
    vote_threshold=None,
):
    """
    N-modal reliable-neighbor voting.

    For M modalities, a candidate j for center i is supported by every view
    whose KNN_i contains j.  For the TEA-seq three-modality experiment we use
    a 2-of-3 rule instead of strict 3-way intersection:

        support(i,j) >= 2

    Priority:
      1) vote-mutual: supported by >= threshold views AND mutual in
         >= threshold views;
      2) vote-consensus: supported by >= threshold views;
      3) rank-union fallback when no consensus candidate exists.

    For M=2, vote_threshold defaults to 2, so this reduces naturally to the
    original all-view consensus logic.
    """
    if len(knn_indices_by_view) < 2:
        raise ValueError("Reliable MNN requires at least two modalities.")

    n_views = len(knn_indices_by_view)
    n = int(knn_indices_by_view[0].shape[0])
    k = int(knn_indices_by_view[0].shape[1])

    if vote_threshold is None:
        vote_threshold = max(2, n_views // 2 + 1)
    vote_threshold = int(vote_threshold)

    if vote_threshold < 1 or vote_threshold > n_views:
        raise ValueError(
            "vote_threshold must be between 1 and the number of modalities."
        )

    knn_sets = []
    rank_maps = []

    for knn in knn_indices_by_view:
        if int(knn.shape[0]) != n or int(knn.shape[1]) != k:
            raise ValueError("All modality KNN matrices must share one shape.")

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
                    for rank, j in enumerate(knn[i].tolist())
                }
            )
        rank_maps.append(ranks)

    selected_lists = []
    count_vote_mutual = 0
    count_vote_consensus = 0
    count_fallback = 0
    total_selected = 0

    for i in range(n):
        union = set.union(*[knn_sets[v][i] for v in range(n_views)])

        candidates = []
        for j in union:
            j = int(j)
            if j == i:
                continue

            support = sum(
                1 for v in range(n_views)
                if j in knn_sets[v][i]
            )
            mutual_support = sum(
                1 for v in range(n_views)
                if (
                    j in knn_sets[v][i]
                    and i in knn_sets[v][j]
                )
            )
            avg_rank = float(
                np.mean(
                    [
                        rank_maps[v][i].get(j, k)
                        for v in range(n_views)
                    ]
                )
            )
            candidates.append(
                (j, support, mutual_support, avg_rank)
            )

        mutual_candidates = [
            item for item in candidates
            if (
                item[1] >= vote_threshold
                and item[2] >= vote_threshold
            )
        ]

        if len(mutual_candidates) >= int(min_keep):
            mutual_candidates.sort(
                key=lambda x: (-x[2], -x[1], x[3], x[0])
            )
            selected = [x[0] for x in mutual_candidates[:k]]
            count_vote_mutual += 1
        else:
            consensus_candidates = [
                item for item in candidates
                if item[1] >= vote_threshold
            ]
            if len(consensus_candidates) > 0:
                consensus_candidates.sort(
                    key=lambda x: (-x[1], -x[2], x[3], x[0])
                )
                selected = [x[0] for x in consensus_candidates[:k]]
                count_vote_consensus += 1
            else:
                candidates.sort(
                    key=lambda x: (-x[1], -x[2], x[3], x[0])
                )
                selected = [
                    x[0]
                    for x in candidates[:max(1, int(min_keep))]
                ]
                count_fallback += 1

        if len(selected) == 0:
            selected = [int(knn_indices_by_view[0][i, 0])]

        selected_lists.append(selected)
        total_selected += len(selected)

    max_len = max(len(values) for values in selected_lists)
    neighbor_ids = np.zeros((n, max_len), dtype=np.int64)
    neighbor_mask = np.zeros((n, max_len), dtype=np.float32)

    for i, selected in enumerate(selected_lists):
        length = len(selected)
        neighbor_ids[i, :length] = np.asarray(selected, dtype=np.int64)
        neighbor_mask[i, :length] = 1.0

    print(
        "N-modal MNN stats: k={}, views={}, vote_threshold={}, "
        "mean_selected={:.4f}, vote_mutual_ratio={:.6f}, "
        "vote_consensus_ratio={:.6f}, fallback_ratio={:.6f}".format(
            k,
            n_views,
            vote_threshold,
            total_selected / float(max(n, 1)),
            count_vote_mutual / float(max(n, 1)),
            count_vote_consensus / float(max(n, 1)),
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
    N-modal pairwise pure-geometric UOT.

    For M modalities, construct all M(M-1)/2 pairwise couplings in the
    modality-specific latent spaces.  A pair (a,b) contributes a relation-
    induced neighbor direction to both modality a and modality b.

    Importantly, cross-modal geometry chooses CELL INDICES, but aggregation
    remains inside the target modality's own feature space.  Therefore RNA,
    ATAC and ADT feature dimensions do not need to match.

    For view v:
        shift_v = lambda_uot *
                  mean_{u != v}(neighbor_{v <- u} - x_v)

    Thus the total update scale stays comparable with the original bimodal
    lambda=0.15 instead of doubling when a third modality is introduced.

    Feature Gate is intentionally OFF inside UOT.
    Mass Gate is intentionally OFF.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    centers, valid_indices = _get_center_inputs(data)
    n_views = len(centers)

    if n_views < 2:
        raise ValueError("N-modal UOT requires at least two modalities.")

    n_cells = int(centers[0].shape[0])
    if any(int(x.shape[0]) != n_cells for x in centers):
        raise ValueError(
            "Current paired N-modal UOT requires equal cell counts in all views."
        )

    view_latents, _ = _compute_view_latents_from_pretrained_vae(
        vae,
        data,
        device=device,
        batch_size=latent_batch_size,
    )
    if view_latents is None or len(view_latents) != n_views:
        raise RuntimeError("Failed to compute all modality latent matrices.")

    z_all = [
        F.normalize(z.float(), p=2, dim=1)
        for z in view_latents
    ]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 5151)
    permutation = torch.randperm(
        n_cells,
        generator=generator,
    )

    shift_sums = [0.0 for _ in range(n_views)]
    total_cells = 0
    num_batches = 0
    pair_count = n_views * (n_views - 1) // 2

    with torch.no_grad():
        for batch_start in range(
            0,
            n_cells,
            int(uot_batch_size),
        ):
            indices = permutation[
                batch_start:
                batch_start + int(uot_batch_size)
            ]
            current_size = int(indices.numel())

            if current_size < 2:
                continue

            # Read all original center features for this batch before any view
            # is modified.  Every cell appears in exactly one deterministic
            # mini-batch, so committing after all pair calculations is safe.
            x_batch = [
                centers[v][indices].to(device).float()
                for v in range(n_views)
            ]
            z_batch = [
                z_all[v][indices].to(device)
                for v in range(n_views)
            ]

            accumulated_direction = [
                torch.zeros_like(x_batch[v])
                for v in range(n_views)
            ]
            partner_counts = [0 for _ in range(n_views)]

            for a in range(n_views):
                for b in range(a + 1, n_views):
                    cost = torch.cdist(
                        z_batch[a],
                        z_batch[b],
                        p=2,
                    ).pow(2)

                    cost_min = cost.min()
                    cost_max = cost.max()
                    cost = (
                        (cost - cost_min)
                        / (cost_max - cost_min + 1e-12)
                    )

                    if exclude_paired_diagonal:
                        diag = torch.arange(
                            current_size,
                            device=device,
                        )
                        cost[diag, diag] = 1.5

                    coupling = _unbalanced_sinkhorn(
                        cost,
                        entropic_reg=entropic_reg,
                        mass_reg=mass_reg,
                        max_iter=sinkhorn_iter,
                    )

                    # a -> b cell relation, applied inside modality a features
                    row_weights = _topk_row_normalize(
                        coupling,
                        top_k=uot_top_k,
                    )
                    # b -> a cell relation, applied inside modality b features
                    col_weights = _topk_row_normalize(
                        coupling.transpose(0, 1),
                        top_k=uot_top_k,
                    )

                    neighbor_a = torch.matmul(
                        row_weights,
                        x_batch[a],
                    )
                    neighbor_b = torch.matmul(
                        col_weights,
                        x_batch[b],
                    )

                    accumulated_direction[a].add_(
                        neighbor_a - x_batch[a]
                    )
                    accumulated_direction[b].add_(
                        neighbor_b - x_batch[b]
                    )
                    partner_counts[a] += 1
                    partner_counts[b] += 1

            for v in range(n_views):
                if partner_counts[v] <= 0:
                    continue

                shift_v = (
                    float(lambda_uot)
                    * accumulated_direction[v]
                    / float(partner_counts[v])
                )
                new_v = x_batch[v] + shift_v

                data[valid_indices[v]][
                    indices,
                    :,
                    0,
                ] = new_v.detach().cpu().to(
                    data[valid_indices[v]].dtype
                )

                shift_sums[v] += float(
                    torch.norm(
                        shift_v,
                        p=2,
                        dim=1,
                    ).sum().detach().cpu()
                )

            total_cells += current_size
            num_batches += 1

    mean_shifts = [
        value / float(max(total_cells, 1))
        for value in shift_sums
    ]

    print(
        "N-modal pairwise UOT complete: views={}, pairs={}, "
        "lambda={:.2f}, top_k={}, batches={}, "
        "feature_gate=False, mass_gate=False".format(
            n_views,
            pair_count,
            lambda_uot,
            int(uot_top_k),
            num_batches,
        )
    )
    print(
        "N-modal UOT mean shifts per view:",
        [round(v, 6) for v in mean_shifts],
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
        "M3: SP-GeDGC + Latent SpectralNet "
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
    trajectory_path = result_file("Trajectory_latent_SP_LSNTA_GeDGC.npz")
    np.savez_compressed(
        trajectory_path,
        latent=x_mean.detach().cpu().numpy(),
        y_true=np.asarray(label),
    )
    print("Saved trajectory latent ->", trajectory_path)

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
        "| FINAL N-MODAL ACC = {:.6f} NMI = {:.6f} "
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
