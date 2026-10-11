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
            encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder.append(nn.BatchNorm1d(layer_sizes[i + 1], affine=True))
            encoder.append(activation)
        for i in range(self.depth - 1):
            decoder.append(nn.Linear(layer_sizes[self.depth - i], layer_sizes[self.depth - i - 1]))
            decoder.append(nn.BatchNorm1d(layer_sizes[self.depth - i - 1], affine=True))
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
    """
    SP-GeDGC: Shared-Private-only modification of the original GeDGC Multi-view VAE.

    Everything used for clustering remains on the original shared latent path:
        z_shared = weighted fusion of modality latent distributions

    The only modification is the reconstruction input of each modality:
        z_private_v = z_view_mean - z_shared_mean
        z_recon_v   = z_sample + private_beta * z_private_v

    Default:
        private_beta = 0.1

    No neighbor aggregation, Leiden, MNN, UOT, causal module, agent,
    residual selector, uncertainty fusion, prototype module, or other
    enhancement is included here.
    """

    def __init__(self, layer_sizes, activation=nn.ReLU(), private_beta=0.1):
        super(Multiview_VAE, self).__init__()

        # Original GeDGC learnable modality fusion weights.
        self.w = nn.Parameter(torch.ones(len(layer_sizes)) / len(layer_sizes))

        # Original per-modality VAE branches.
        self.vaes = nn.ModuleList(
            [vae(layer_size, activation) for layer_size in layer_sizes]
        )

        # Only new hyperparameter in this patch.
        self.private_beta = float(private_beta)

        # Cache the current per-view means and shared mean.
        # The original pretrain.py/train.py call get_latent() immediately
        # before get_recon(), so no external interface needs to change.
        self._last_view_means = None
        self._last_shared_mean = None

    def get_view_latent(self, inputs):
        """
        Return modality-specific latent distributions before shared fusion.
        This helper does not change the original GeDGC downstream interface.
        """
        view_means = []
        view_logvars = []

        for view in range(len(inputs)):
            mean, logvar = self.vaes[view].get_latent(inputs[view])
            view_means.append(mean)
            view_logvars.append(logvar)

        return view_means, view_logvars

    def get_latent(self, inputs):
        """
        Original GeDGC shared latent fusion.

        Clustering, GMM, classifier and graph constraints still receive exactly
        this shared latent distribution.
        """
        x_mean = 0
        x_var = 0

        # Keep the original GeDGC fusion formula unchanged.
        w = torch.exp(self.w) / torch.sum(torch.exp(self.w))

        view_means = []
        view_logvars = []

        for view in range(len(inputs)):
            mean, logvar = self.vaes[view].get_latent(inputs[view])

            view_means.append(mean)
            view_logvars.append(logvar)

            x_mean = x_mean + mean * w[view]
            x_var = x_var + torch.pow(
                torch.exp(0.5 * logvar) * w[view],
                2,
            )

        # Cache only for the SP reconstruction path.
        self._last_view_means = view_means
        self._last_shared_mean = x_mean

        return x_mean, torch.log(x_var + 1e-10)

    def get_recon(self, inputs):
        """
        Shared-private reconstruction.

        Original GeDGC:
            every decoder receives the same shared latent sample.

        SP-GeDGC:
            decoder_v receives
            inputs + private_beta * (z_view_mean - z_shared_mean)

        If no matching latent cache exists, this safely falls back to the
        original GeDGC reconstruction path.
        """
        recon = []

        use_private = (
            self._last_view_means is not None
            and self._last_shared_mean is not None
            and len(self._last_view_means) == len(self.vaes)
            and self._last_shared_mean.shape == inputs.shape
        )

        for view in range(len(self.vaes)):
            if use_private:
                z_private = (
                    self._last_view_means[view]
                    - self._last_shared_mean
                )
                recon_input = (
                    inputs
                    + self.private_beta * z_private
                )
            else:
                recon_input = inputs

            recon.append(
                self.vaes[view].get_recon(recon_input)
            )

        return recon

    def forward(self):
        pass
