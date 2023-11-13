import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
from layers import Causal_GraphConvolution,GraphConvolution
from torch.nn.parameter import Parameter


class GCNModelSIGVAE(nn.Module):
    def __init__(self, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto='semi', gdc='ip', ndist = 'Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(GCNModelSIGVAE, self).__init__()

        self.gce = Causal_GraphConvolution(ndim, hidden_dim1, dropout, act=F.relu) #hiddene = self.gce(e, adj)
        self.gc1 = Causal_GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu) # hiddenx = self.gc1(x, adj)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x) # mu = self.gc2(hidden1, adj)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x) # logvar = self.gc3(hidden_sd, adj)
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        if ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':
            self.ndist == tdist.Normal(
                    torch.tensor([0.], device=self.device),
                    torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        # K and J are defined in http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # Algorthm 1.
        self.K = copyK
        self.J = copyJ
        self.ndim = ndim

        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output 
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).
        self.reweight = ((self.ndim + hidden_dim1) / (input_feat_dim + hidden_dim1))**(.5)


    def encode(self, x, adj):
        assert len(x.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.

        hiddenx = self.gc1(x, adj)


        if self.ndim >= 1:
            e = self.ndist.sample(torch.Size([self.K+self.J, x.shape[1], self.ndim]))
            e = torch.squeeze(e, -1)
            e = e.mul(self.reweight)
            hiddene = self.gce(e, adj)

        else:
            print("no randomness.")  # '--edim', type=int, default=32, help='Number of units in noise epsilon.')
            hiddene = torch.zeros(self.K+self.J, hiddenx.shape[1], hiddenx.shape[2], device=self.device)

        
        hidden1 = hiddenx + hiddene

        # hiddens = self.gc0(x, adj)

        p_signal = hiddenx.pow(2.).mean()
        p_noise = hiddene.pow(2.).mean([-2,-1])
        snr = (p_signal / p_noise)
        

        # below are 3 options for producing logvar
        # 1. stochastic logvar (more instinctive)
        #    where logvar = self.gc3(hidden1, adj)
        #    set args.encsto to 'full'.
        # 2. deterministic logvar, shared by all K+J samples, and share a previous hidden layer with mu
        #    where logvar = self.gc3(hiddenx, adj)
        #    set args.encsto to 'semi'.
        # 3. deterministic logvar, shared by all K+J samples, and produced by another branch of network
        #    (the one applied by A. Hasanzadeh et al.)
        

        mu = self.gc2(hidden1, adj)

        EncSto = (self.encsto == 'full') # encsto='semi' #EncSto False
        hidden_sd = EncSto * hidden1 + (1 - EncSto) * hiddenx

        logvar = self.gc3(hidden_sd, adj)

        return mu, logvar, snr

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps
        # return mu, eps

    def forward(self, x, adj):
        mu, logvar, snr = self.encode(x, adj)
        
        emb_mu = mu[self.K:, :]
        emb_logvar = logvar[self.K:, :]

        # check tensor size compatibility
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'

        z, eps = self.reparameterize(emb_mu, emb_logvar)

        adj_, z_scaled, rk = self.dc(z)

        return adj_, mu, logvar, z, z_scaled, eps, rk, snr


class GraphDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        assert self.zdim == z.shape[2], 'zdim not compatible!'

        # The variable 'rk' in the code is the square root of the same notation in
        # http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # i.e., instead of do Z*diag(rk)*Z', we perform [Z*diag(rk)] * [Z*diag(rk)]'.
        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        # Z shape: [J, N, zdim]
        # Z' shape: [J, zdim, N]
        if self.gdc == 'bp':
            z = z.mul(rk.view(1, 1, self.zdim))
        adj_lgt = torch.bmm(z, torch.transpose(z, 1, 2))

        if self.gdc == 'ip':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            # 1 - exp( - exp(ZZ'))
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())


        # if self.training:
        #     adj_lgt = - torch.log(1 / (adj + self.SMALL) - 1 + self.SMALL)   
        # else:
        #     adj_mean = torch.mean(adj, dim=0, keepdim=True)
        #     adj_lgt = - torch.log(1 / (adj_mean + self.SMALL) - 1 + self.SMALL)

        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)
        
        return adj, z, rk.pow(2)