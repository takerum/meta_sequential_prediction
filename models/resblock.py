import sys
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.weight_standarization import WeightStandarization, WeightStandarization1d
import torch.nn.utils.parametrize as P
from utils.emb2d import Emb2D


def upsample_conv(x, conv):
    # Upsample -> Conv
    x = nn.Upsample(scale_factor=2, mode='nearest')(x)
    x = conv(x)
    return x


def conv_downsample(x, conv):
    # Conv -> Downsample
    x = conv(x)
    h = F.avg_pool2d(x, 2)
    return h


class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 kernel_size=3,
                 padding=None,
                 activation=F.relu,
                 resample=None,
                 group_norm=True,
                 skip_connection=True,
                 posemb=False):
        super(Block, self).__init__()
        if padding is None:
            padding = (kernel_size-1) // 2
        self.pe = Emb2D() if posemb else lambda x: x

        in_ch_conv = in_channels + self.pe.dim if posemb else in_channels
        self.skip_connection = skip_connection
        self.activation = activation
        self.resample = resample
        initializer = torch.nn.init.xavier_uniform_
        if self.resample is None or self.resample == 'up':
            hidden_channels = out_channels if hidden_channels is None else hidden_channels
        else:
            hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_ch_conv, hidden_channels,
                            kernel_size=kernel_size, padding=padding)
        self.c2 = nn.Conv2d(hidden_channels, out_channels,
                            kernel_size=kernel_size, padding=padding)
        initializer(self.c1.weight, math.sqrt(2))
        initializer(self.c2.weight, math.sqrt(2))
        P.register_parametrization(
            self.c1, 'weight', WeightStandarization())
        P.register_parametrization(
            self.c2, 'weight', WeightStandarization())

        if group_norm:
            self.b1 = nn.GroupNorm(min(32, in_channels), in_channels)
            self.b2 = nn.GroupNorm(min(32, hidden_channels), hidden_channels)
        else:
            self.b1 = self.b2 = lambda x: x
        if self.skip_connection:
            self.c_sc = nn.Conv2d(in_ch_conv, out_channels,
                                  kernel_size=1, padding=0)
            initializer(self.c_sc.weight)

    def residual(self, x):
        x = self.b1(x)
        x = self.activation(x)
        if self.resample == 'up':
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.pe(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.activation(x)
        x = self.c2(x)
        if self.resample == 'down':
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        # Upsample -> Conv
        if self.resample == 'up':
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            x = self.pe(x)
            x = self.c_sc(x)

        elif self.resample == 'down':
            x = self.pe(x)
            x = self.c_sc(x)
            x = F.avg_pool2d(x, 2)
        else:
            x = self.pe(x)
            x = self.c_sc(x)
        return x

    def __call__(self, x):
        if self.skip_connection:
            return self.residual(x) + self.shortcut(x)
        else:
            return self.residual(x)


class Conv1d1x1Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 act=F.relu):
        super().__init__()

        self.act = act
        initializer = torch.nn.init.xavier_uniform_
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)
        self.c2 = nn.Conv1d(hidden_channels, out_channels, 1, 1, 0)
        initializer(self.c1.weight, math.sqrt(2))
        initializer(self.c2.weight, math.sqrt(2))
        P.register_parametrization(
            self.c1, 'weight', WeightStandarization1d())
        P.register_parametrization(
            self.c2, 'weight', WeightStandarization1d())
        self.norm1 = nn.LayerNorm((in_channels))
        self.norm2 = nn.LayerNorm((hidden_channels))
        self.c_sc = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        initializer(self.c_sc.weight)

    def residual(self, x):
        x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.c1(x)
        x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.c2(x)
        return x

    def shortcut(self, x):
        x = self.c_sc(x)
        return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)