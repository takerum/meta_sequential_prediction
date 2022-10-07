import math
import numpy as np
import torch
import torch.nn as nn


class Emb2D(nn.modules.lazy.LazyModuleMixin, nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.emb = torch.nn.parameter.UninitializedParameter()

    def __call__(self, x):
        if torch.nn.parameter.is_lazy(self.emb):
            _, h, w = x.shape[1:]
            self.emb.materialize((self.dim, h, w))
            self.emb.data = positionalencoding2d(self.dim, h, w)
        emb = torch.tile(self.emb[None].to(x.device), [x.shape[0], 1, 1, 1])
        x = torch.cat([x, emb], axis=1)
        return x

# Copied from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
