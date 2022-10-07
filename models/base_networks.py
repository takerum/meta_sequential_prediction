import numpy as np
import torch
from torch import nn
from models.resblock import Block, Conv1d1x1Block
from einops.layers.torch import Rearrange
from einops import repeat


class Conv1d1x1Encoder(nn.Sequential):
    def __init__(self,
                 dim_out=16,
                 dim_hidden=128,
                 act=nn.ReLU()):
        super().__init__(
            nn.LazyConv1d(dim_hidden, 1, 1, 0),
            Conv1d1x1Block(dim_hidden, dim_hidden, act=act),
            Conv1d1x1Block(dim_hidden, dim_hidden, act=act),
            Rearrange('n c s -> n s c'),
            nn.LayerNorm((dim_hidden)),
            Rearrange('n s c-> n c s'),
            act,
            nn.LazyConv1d(dim_out, 1, 1, 0)
        )


class ResNetEncoder(nn.Module):
    def __init__(self,
                 dim_latent=1024,
                 k=1,
                 act=nn.ReLU(),
                 kernel_size=3,
                 n_blocks=3):
        super().__init__()
        self.phi = nn.Sequential(
            nn.LazyConv2d(int(32 * k), 3, 1, 1),
            *[Block(int(32 * k) * (2 ** i), int(32 * k) * (2 ** (i+1)), int(32 * k) * (2 ** (i+1)),
                    resample='down', activation=act, kernel_size=kernel_size) for i in range(n_blocks)],
            nn.GroupNorm(min(32, int(32 * k) * (2 ** n_blocks)),
                         int(32 * k) * (2 ** n_blocks)),
            act)
        self.linear = nn.LazyLinear(
            dim_latent) if dim_latent > 0 else lambda x: x

    def __call__(self, x):
        h = x
        h = self.phi(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h


class ResNetDecoder(nn.Module):
    def __init__(self, ch_x, k=1, act=nn.ReLU(), kernel_size=3, bottom_width=4, n_blocks=3):
        super().__init__()
        self.bottom_width = bottom_width
        self.linear = nn.LazyLinear(int(32 * k) * (2 ** n_blocks))
        self.net = nn.Sequential(
            *[Block(int(32 * k) * (2 ** (i+1)), int(32 * k) * (2 ** i), int(32 * k) * (2 ** i),
                    resample='up', activation=act, kernel_size=kernel_size, posemb=True) for i in range(n_blocks-1, -1, -1)],
            nn.GroupNorm(min(32, int(32 * k)), int(32 * k)),
            act,
            nn.Conv2d(int(32 * k), ch_x, 3, 1, 1)
        )

    def __call__(self, x):
        x = self.linear(x)
        x = repeat(x, 'n c -> n c h w',
                   h=self.bottom_width, w=self.bottom_width)
        x = self.net(x)
        x = torch.sigmoid(x)
        return x
