import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from tools.utils import cluster_acc, purity, to_numpy
from tools.early_stopping import EarlyStopping
from tqdm import trange
from tools.load_data import Cell

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def gen_x(mean, std, J):
    x_samples = []
    v_size = mean.size()
    for idx in range(J):
        x_samples.append(mean + torch.mul(std, torch.randn(v_size).cuda()))

    return x_samples


def compute_weight(inputs, similarity_type='Gauss'):
    dist = 0
    for input_ in inputs:
        dist += torch.sum(torch.pow(input_ - input_[:, :, 0].unsqueeze(2), 2), dim=1)
    dist = F.normalize(dist, dim=1)

    if similarity_type == 'Gauss':
        Gauss_simi = torch.exp(-dist)
        if Gauss_simi.shape[1] == 1:
            Gauss_simi[:, 0] = 1
        else:
            Gauss_simi[:, 0] = torch.sum(Gauss_simi[:, 1:], dim=1)
        simi = torch.div(Gauss_simi, torch.sum(Gauss_simi, dim=1, keepdim=True))
    else:
        N = inputs[0].size(-1)
        simi = torch.ones(1, N) / (N - 1)
        simi[0, 0] = 1
        simi = torch.mul(torch.ones(inputs[0].size(0), 1), simi)
        simi = torch.div(simi, torch.sum(simi, dim=1, keepdim=True))

    return simi


def train_model(vae, classifier, GMM, optimizer, lr_scheduler, dataloader, dataname, epoch_num, device):
    vae = vae.to(device)
    classifier = classifier.to(device)
    GMM = GMM.to(device)

    avg_loss = []
    early_stopping = EarlyStopping(patience=10, delta=1e-3)

    t = trange(epoch_num, leave=True)

    for epoch in t:
        Total_loss = []
        label = []
        label_pred = []
        Recon_loss = []

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            N_samples = inputs[0].size(0)
            N_n = inputs[0].size(2)

            inputs = [input_.to(device) for input_ in inputs]

            all_inputs = []
            all_targets = []
            for data_v in inputs:
                temp_data1 = []
                temp_data2 = []
                for idx in range(N_n):
                    temp_data1.append(data_v[:, :, idx])
                    temp_data2.append(data_v[:, :, 0])

                temp_data1 = torch.cat(temp_data1, dim=0)
                temp_data2 = torch.cat(temp_data2, dim=0)

                all_inputs.append(temp_data1)
                all_targets.append(temp_data2)

            label.append(labels)

            weight = compute_weight(inputs)
            weight_temp = []
            for idx in range(N_n):
                weight_temp.append(weight[:, idx])
            weight = torch.cat(weight_temp)

            # Compute the prior of c
            vae.eval()
            classifier.eval()
            with torch.no_grad():
                x_mean, _ = vae.get_latent(all_inputs)
                pc = classifier(x_mean).data

            pc = torch.mul(pc, weight.view(-1, 1))
            pc_temp = 0
            for idx in range(N_n):
                pc_temp = pc_temp + pc[idx * N_samples:(idx + 1) * N_samples, :]
            pc_temp1 = []
            for idx in range(N_n):
                pc_temp1.append(pc_temp)

            pc = torch.cat(pc_temp1, dim=0)

            # Begin training
            vae.train()
            classifier.train()
            GMM.train()

            loss = 0

            J = 1
            x_mean, x_logvar = vae.get_latent(all_inputs)
            x_samples = gen_x(x_mean, torch.exp(0.5 * x_logvar), J)
            ELBO = 0
            for idx in range(J):
                x_re = vae.get_recon(x_samples[idx])
                # x_re = vae.get_recon(torch.cat((x_samples[idx], dbatch), dim=1))

                recon = 0
                for view in range(len(x_re)):
                    recon += -0.5 * torch.sum(torch.pow(all_targets[view] - x_re[view], 2), dim=1)

                ELBO = ELBO + 0.1*torch.sum(torch.mul(recon, weight))

            ELBO = ELBO / J
            # print(ELBO)
            Recon_loss.append((-ELBO / N_samples).item())

            cond_prob = classifier(x_samples[0])
            ELBO = ELBO + torch.sum(
                torch.mul(torch.sum(-torch.mul(cond_prob, torch.log(cond_prob + 1e-10)), dim=-1), weight))
            ELBO = ELBO + torch.sum(
                torch.mul(torch.sum(torch.mul(cond_prob, torch.log(pc + 1e-10)), dim=-1), weight))
            ELBO = ELBO + GMM.log_prob(x_mean, x_logvar, cond_prob, weight)
            ELBO = ELBO + torch.sum(torch.mul(0.5 * torch.sum(x_logvar, dim=-1), weight))

            loss = loss - ELBO / N_samples
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_loss.append(loss.item())
            
            if np.isnan(loss.item()):
                return vae, classifier, GMM, avg_loss

            vae.eval()
            classifier.eval()
            data_i = []
            for data_v in inputs:
                data_i.append(data_v[:, :, 0].data)
            with torch.no_grad():
                x_mean, _ = vae.get_latent(data_i)
                pred_l = torch.max(classifier(x_mean), dim=-1)
            label_pred.append(pred_l[-1])

        lr_scheduler.step()

        label = torch.cat(label, dim=0)
        label_pred = torch.cat(label_pred, dim=0)

        NMI = normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred))
        ARI = adjusted_rand_score(to_numpy(label), to_numpy(label_pred))
        pur = purity(to_numpy(label), to_numpy(label_pred))
        acc, _ = cluster_acc(to_numpy(label_pred), to_numpy(label))

        avg_loss.append(np.mean(Total_loss))

        t.set_description('|Epoch:{} Total loss={:3f} Reconstruction Loss={:6f} PUR={:5f} ARI={:5f} NMI={:5f} ACC={:6f}'.format(
                epoch + 1, np.mean(Total_loss), np.mean(Recon_loss), pur, ARI, NMI, acc))
        t.refresh()

        early_stopping(np.mean(Total_loss))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return vae, classifier, GMM, avg_loss

def train_opt(data, label, dataname):
    if data[0].shape[0] > 1024:
        batch_size = 128
    else:
        batch_size = 16
    learning_rate = 0.0001
    weight_decay = 1e-6
    step_size = 50
    gama = 0.1
    epoch = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Load pretrained model")
    vae = torch.load('Intermediate_data/' + dataname + '/pretrained_vae.pkl')
    classifier = torch.load('Intermediate_data/' + dataname + '/pretrained_classifier.pkl')
    GMM = torch.load('Intermediate_data/' + dataname + '/pretrained_GMM.pkl')

    print("Train the model")
    dataloader = DataLoader(Cell(data, label), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(list(vae.parameters()) + list(classifier.parameters()) + list(GMM.parameters()),
                           lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gama)

    vae, classifier, GMM, avg_loss = train_model(vae, classifier, GMM, optimizer, lr_scheduler, dataloader, dataname, epoch, device)

    vae.eval()
    classifier.eval()
    GMM.eval()

    datai = [data_[:, :, 0] for data_ in data]
    dataloader = DataLoader(Cell(datai), batch_size=1024, shuffle=False)
    x_mean = []
    x_logvar = []
    for batch_idx, (inputs) in enumerate(dataloader):
        inputs = [input_.to(device) for input_ in inputs]
        with torch.no_grad():
            x_m, x_v = vae.get_latent(inputs)
        x_mean.append(x_m)
        x_logvar.append(x_v)
    x_mean = torch.cat(x_mean, dim=0)
    x_logvar = torch.cat(x_logvar, dim=0)
    # === TRAJECTORY_LATENT_EXPORT ===
    import os as _traj_os
    import numpy as _traj_np

    _traj_dir = _traj_os.path.join("result", dataname)
    _traj_os.makedirs(_traj_dir, exist_ok=True)
    _traj_path = _traj_os.path.join(
        _traj_dir,
        "Trajectory_latent_GeDGC.npz",
    )
    _traj_np.savez_compressed(
        _traj_path,
        latent=x_mean.detach().cpu().numpy(),
        y_true=_traj_np.asarray(label),
    )
    print("Saved trajectory latent ->", _traj_path)

    x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar), 1)[0]
    with torch.no_grad():
        pred_label = torch.max(classifier(x_sample), dim=-1)
    acc, _ = cluster_acc(to_numpy(pred_label[-1]), label)
    NMI = normalized_mutual_info_score(label, to_numpy(pred_label[-1]))
    ARI = adjusted_rand_score(label, to_numpy(pred_label[-1]))
    pur = purity(label, to_numpy(pred_label[-1]))
    FMI = fowlkes_mallows_score(label, to_numpy(pred_label[-1]))    

    # ------------------------------------------------------------
    # UMAP export: use the deterministic posterior mean x_mean
    # rather than the sampled x_sample. This is only for visualization
    # and does not change any training/evaluation result.
    # ------------------------------------------------------------
    os.makedirs('result/' + dataname, exist_ok=True)
    np.savez_compressed(
        'result/' + dataname + '/UMAP_latent_GeDGC.npz',
        latent=to_numpy(x_mean),
        y_true=np.asarray(label),
        y_pred=to_numpy(pred_label[-1]),
    )
    print('Saved UMAP latent -> result/' + dataname + '/UMAP_latent_GeDGC.npz')

    vae = vae.to('cpu')
    classifier = classifier.to('cpu')
    GMM = GMM.to('cpu')

    torch.save(vae, 'result/' + dataname + '/trained_vae.pkl')
    torch.save(classifier, 'result/' + dataname + '/trained_classifier.pkl')
    torch.save(GMM, 'result/' + dataname + '/trained_GMM.pkl')

    return acc, NMI, ARI, pur, FMI
