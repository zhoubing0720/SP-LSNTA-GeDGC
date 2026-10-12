import torch
import torch.nn as nn
import math

class Classifier(nn.Module):
    def __init__(self,layer_sizes,activation=nn.ReLU()):
        super(Classifier,self).__init__()
        n_layer = len(layer_sizes)
        layers = []
        for idx in range(n_layer-1):
            layer = nn.Linear(layer_sizes[idx],layer_sizes[idx+1])
            nn.init.xavier_normal_(layer.weight)
            # layer.bias.data = torch.zeros(layer_sizes[idx + 1])
            layers.append(layer)
            if idx < n_layer-2:
                layers.append(activation)
            else:
                layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self,inputs):
        output = self.model(inputs)
        return output


class GMM_Model(nn.Module):
    def __init__(self, N, K, mean=None, var=None):
        super(GMM_Model, self).__init__()
        if mean is not None:
            self.mean = nn.Parameter(mean.view(1, N, K))
            self.std = nn.Parameter(torch.sqrt(var).view(1, N, K))
        else:
            self.mean = nn.Parameter(torch.randn(1, N, K))
            self.std = nn.Parameter(torch.ones(1, N, K))

        self.N = N # dim
        self.K = K # label

    def compute_prob(self, data):
        prob = torch.exp(torch.sum(
            -torch.log((self.std ** 2) * 2 * math.pi) - torch.div(torch.pow(data.view(-1, self.N, 1) - self.mean, 2),
                                                                  self.std ** 2), dim=1) * 0.5)
        pc = torch.div(prob, (torch.sum(prob, dim=-1)).view(-1, 1) + 1e-10)
        return pc

    def log_prob(self, data_mean, data_logvar, cond_prob, weight):
        # term1 = torch.sum(-torch.log((self.std ** 2) * 2 * math.pi), dim=1) * 0.5
        term1 = torch.sum(-torch.log(self.std ** 2), dim=1) * 0.5
        term2 = torch.sum(-torch.div(
            torch.pow(data_mean.view(-1, self.N, 1) - self.mean, 2) + torch.exp(data_logvar).view(-1, self.N, 1),
            self.std ** 2), dim=1) * 0.5
        prob = term2 + term1
        log_p1 = torch.sum(torch.mul(prob, cond_prob), dim=-1)
        log_p = torch.sum(torch.mul(log_p1, weight))

        return log_p

    def forward(self):
        pass