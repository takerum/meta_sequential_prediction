import torch
import torch.nn as nn
from models.base_networks import ResNetEncoder
from einops import rearrange


class ResNetwProjHead(nn.Module):

    def __init__(self, dim_mlp=512, dim_head=128, k=1, act=nn.ReLU(), n_blocks=3):
        super().__init__()
        self.enc = ResNetEncoder(
            dim_latent=0, k=k, n_blocks=n_blocks)
        self.projhead = nn.Sequential(
            nn.LazyLinear(dim_mlp),
            act,
            nn.LazyLinear(dim_head))

    def _encode_base(self, xs, enc):
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)
        H = torch.reshape(H, (shape[0], shape[1], *H.shape[1:]))
        return H

    def __call__(self, xs):
        return self._encode_base(xs, lambda x: self.projhead(self.enc(x)))

    def phi(self, xs):
        return self._encode_base(xs, self.enc.phi)

    def get_M(self, xs):
        T = xs.shape[1]
        xs = rearrange(xs, 'n t c h w -> (n t) c h w')
        H = self.enc(xs)
        H = rearrange(H, '(n t) c -> n (t c)', t=T)
        return H
