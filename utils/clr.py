import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_pfn_extras as ppe


def simclr(zs, temperature=1.0, normalize=True, loss_type='cossim'):
    if normalize:
        zs = [F.normalize(z, p=2, dim=1) for z in zs]
    m = len(zs)
    n = zs[0].shape[0]
    device = zs[0].device
    mask = torch.eye(n * m, device=device)
    label0 = torch.fmod(n + torch.arange(0, m * n, device=device), n * m)
    z = torch.cat(zs, 0)
    if loss_type == 'euclid':
        sim = - torch.cdist(z, z)
    elif loss_type == 'sq':
        sim = - torch.cdist(z, z) ** 2
    elif loss_type == 'cossim':
        sim = torch.matmul(z, z.transpose(0, 1))
    else:
        raise NotImplementedError
    logit_zz = sim / temperature
    logit_zz += mask * -1e8
    loss = nn.CrossEntropyLoss()(logit_zz, label0)
    return loss
