import torch
import torch.nn as nn

class vae(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(vae, self).__init__()
        
        self.input_dim = layer_sizes[0]
        self.out_dim = layer_sizes[-1]
        self.depth = len(layer_sizes) - 1

        encoder = []
        decoder = []
        for i in range(self.depth - 1):
            encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder.append(nn.BatchNorm1d(layer_sizes[i+1], affine=True))
            encoder.append(activation)
        for i in range(self.depth - 1):
            decoder.append(nn.Linear(layer_sizes[self.depth-i], layer_sizes[self.depth-i-1]))
            decoder.append(nn.BatchNorm1d(layer_sizes[self.depth-i-1], affine=True))
            decoder.append(activation)
        decoder.append(nn.Linear(layer_sizes[1], layer_sizes[0]))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.mean = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.logvar = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def get_latent(self, inputs):
        h = self.encoder(inputs)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return mean, logvar
    
    def get_recon(self, inputs):
        recon = self.decoder(inputs)
        return recon


class Multiview_VAE(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(Multiview_VAE, self).__init__()

        self.w = nn.Parameter(torch.ones(len(layer_sizes)) / len(layer_sizes))
        self.vaes = nn.ModuleList([vae(layer_size, activation) for layer_size in layer_sizes])

    def get_latent(self, inputs):
        x_mean = 0
        x_var = 0

        w = torch.exp(self.w) / torch.sum(torch.exp(self.w))

        for view in range(len(inputs)):
            mean, logvar = self.vaes[view].get_latent(inputs[view])
            x_mean += mean * w[view]
            x_var += torch.pow(torch.exp(0.5 * logvar) * w[view], 2)

        return x_mean, torch.log(x_var)

    def get_recon(self, inputs):
        recon = []
        for view in range(len(self.vaes)):
            data = self.vaes[view].get_recon(inputs)
            recon.append(data)
        return recon

    def forward(self):
        pass