import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tools.load_data import Cell
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from module.Multi_vae import Multiview_VAE
from module.classifier import Classifier, GMM_Model
from tools.utils import to_numpy, cluster_acc, purity
from tqdm import trange

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def gen_x(mean, std):
    v_size = mean.size()
    x_samples = mean + torch.mul(std, torch.randn(v_size).cuda())
    return x_samples


def pretrain_vae(vae, optimizer, lr_scheduler, dataloader, epoch_num, device):
    vae = vae.to(device)

    avg_loss = []

    t = trange(epoch_num, leave=True)

    for epoch in t:
        Total_loss = []
        vae.train()
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = [input_.to(device) for input_ in inputs]
            x_mean, x_logvar = vae.get_latent(inputs)
            x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar))
            # x_sample = torch.cat((x_sample, dbatch), dim=1)

            ELBO_rec = 0

            x_re = vae.get_recon(x_sample)

            for view in range(len(x_re)):
                ELBO_rec += 0.5 * F.mse_loss(inputs[view], x_re[view])

            loss = ELBO_rec
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_loss.append(loss.item())

            if np.isnan(loss.item()):
                return vae, avg_loss

        lr_scheduler.step()
        t.set_description('|Epoch:{} Total loss={:3f}'.format(epoch, np.mean(Total_loss)))
        t.refresh()
        avg_loss.append(np.mean(Total_loss))

    return vae, avg_loss


def pretrain_classifier(vae, classifier, optimizer, lr_scheduler, dataloader, epoch_num, device):
    vae.eval()
    classifier = classifier.to(device)
    classifier.train()

    avg_losses = []
    loss_f = nn.NLLLoss()

    t = trange(epoch_num, leave=True)

    for epoch in t:
        Total_loss = []
        label = []
        label_pred = []

        for batch_idx, (inputs, labels, label_train) in enumerate(dataloader):
            inputs = [input_.to(device) for input_ in inputs]
            label_train = label_train.to(device)
            label.append(labels)

            with torch.no_grad():
                x_mean, x_logvar = vae.get_latent(inputs)

            x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar))
            cond_prob = classifier(x_sample)

            loss = loss_f(torch.log(cond_prob + 1e-10), label_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_loss.append(loss.item())

            pred_l = torch.max(cond_prob, dim=-1)
            label_pred.append(pred_l[-1])

            if np.isnan(loss.item()):
                return classifier, avg_losses

        lr_scheduler.step()
        label = torch.cat(label, dim=0)
        label_pred = torch.cat(label_pred, dim=0)
        acc, _ = cluster_acc(to_numpy(label_pred), to_numpy(label))
        t.set_description('|Epoch:{} Total loss={:3f} ACC={:5f}'.format(epoch + 1, np.mean(Total_loss), acc))
        t.refresh()
        avg_losses.append(np.mean(Total_loss))

    return classifier, avg_losses

def pretrain_opt(data, label, c, dataname, train_vae = True):
    if data[0].shape[0] > 1024:
        batch_size = 128
    else:
        batch_size = 16
    learning_rate = 0.001
    weight_decay = 1e-6
    step_size = 50
    gama = 0.1
    epoch = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_vae:

        activation = nn.ReLU()

        encoder_sizes = []
        for _, data_v in enumerate(data):
            encoder_sizes.append([data_v.shape[1], 256, c])

        vae = Multiview_VAE(encoder_sizes, activation=activation)

        dataloader = DataLoader(Cell(data), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gama)

        print("pretrain the VAE model")
        vae, avg_loss = pretrain_vae(vae, optimizer, lr_scheduler, dataloader, epoch, device)

        vae.eval()
        dataloader = DataLoader(Cell(data), batch_size=1024, shuffle=False)
        x_mean = []
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = [input_.to(device) for input_ in inputs]
            with torch.no_grad():
                x_m, _ = vae.get_latent(inputs)
            x_mean.append(x_m)
        x_mean = torch.cat(x_mean, dim=0)
        print("| Latent range: {}/{}".format(x_mean.min(), x_mean.max()))

        kmeans = KMeans(n_clusters=c, random_state=0).fit(to_numpy(x_mean))
        cls_index = kmeans.labels_
        mean = kmeans.cluster_centers_
        mean = torch.from_numpy(mean).to(device)
        acc, _ = cluster_acc(cls_index, label)
        NMI = normalized_mutual_info_score(label, cls_index)
        ARI = adjusted_rand_score(label, cls_index)
        pur = purity(label, cls_index)
        FMI = fowlkes_mallows_score(label, cls_index)
        print('| Kmeans ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f}'.format(acc, NMI, ARI, pur, FMI))
        var = []
        for idx in range(c):
            index = np.where(cls_index == idx)
            var_g = torch.sum((x_mean[index[0], :] - mean[idx, :]) ** 2, dim=0, keepdim=True) / (len(index[0]) - 1)
            var.append(var_g)
        var = torch.cat(var, dim=0)

        GMM = GMM_Model(c, c, mean.t(), var.t())
        GMM = GMM.to(device)
        GMM.eval()
        label_pred = torch.max(GMM.compute_prob(x_mean), dim=-1)
        acc, _ = cluster_acc(to_numpy(label_pred[-1]), label)
        NMI = normalized_mutual_info_score(label, to_numpy(label_pred[-1]))
        ARI = adjusted_rand_score(label, to_numpy(label_pred[-1]))
        pur = purity(label, to_numpy(label_pred[-1]))
        FMI = fowlkes_mallows_score(label, to_numpy(label_pred[-1]))
        print('| GMM ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f}'.format(acc, NMI, ARI, pur, FMI))

        vae = vae.to('cpu')
        GMM = GMM.to('cpu')

        torch.save(vae, 'Intermediate_data/' + dataname + '/pretrained_vae.pkl')
        torch.save(GMM, 'Intermediate_data/' + dataname + '/pretrained_GMM.pkl')
    else:
        vae = torch.load('Intermediate_data/' + dataname + '/pretrained_vae.pkl')
        GMM = torch.load('Intermediate_data/' + dataname + '/pretrained_GMM.pkl')

    vae = vae.to(device)
    GMM = GMM.to(device)
    vae.eval()
    GMM.eval()
    dataloader = DataLoader(Cell(data), batch_size=1024, shuffle=False)
    x_mean = []
    for batch_idx, (inputs) in enumerate(dataloader):
        inputs = [input_.to(device) for input_ in inputs]
        with torch.no_grad():
            x_m, _ = vae.get_latent(inputs)
        x_mean.append(x_m)
    x_mean = torch.cat(x_mean, dim=0)
    with torch.no_grad():
        GMM_label = torch.max(GMM.compute_prob(x_mean), dim=-1)[-1].to('cpu')

    GMM = GMM.to('cpu')

    learning_rate = 0.01
    epoch = 70

    classifier_sizes = [c, c]
    classifier = Classifier(classifier_sizes)

    dataloader = DataLoader(Cell(data, label, GMM_label), batch_size=batch_size,
                            shuffle=True)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gama)

    print("pretrain the classifier")
    classifier, avg_loss = pretrain_classifier(vae, classifier, optimizer, lr_scheduler, dataloader, epoch, device)

    classifier.eval()
    vae.eval()
    dataloader = DataLoader(Cell(data), batch_size=1024, shuffle=False)
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
    x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar))
    with torch.no_grad():
        pred_label = torch.max(classifier(x_sample), dim=-1)
    acc, _ = cluster_acc(to_numpy(pred_label[-1]), label)
    NMI = normalized_mutual_info_score(label, to_numpy(pred_label[-1]))
    ARI = adjusted_rand_score(label, to_numpy(pred_label[-1]))
    pur = purity(label, to_numpy(pred_label[-1]))
    FMI = fowlkes_mallows_score(label, to_numpy(pred_label[-1]))
    print('| classifier ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f}'.format(acc, NMI, ARI, pur, FMI))

    vae = vae.to('cpu')
    classifier = classifier.to('cpu')
    torch.save(classifier, 'Intermediate_data/' + dataname + '/pretrained_classifier.pkl')

    del vae, GMM, classifier

    return 
