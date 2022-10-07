import numpy as np
import torch
import torch.nn as nn
from utils.laplacian import make_identity_like, tracenorm_of_normalized_laplacian, make_identity, make_diagonal
import einops
import pytorch_pfn_extras as ppe


def _rep_M(M, T):
    return einops.repeat(M, "n a1 a2 -> n t a1 a2", t=T)


def _loss(A, B):
    return torch.sum((A-B)**2)


def _solve(A, B):
    ATA = A.transpose(-2, -1) @ A
    ATB = A.transpose(-2, -1) @ B
    return torch.linalg.solve(ATA, ATB)


def loss_bd(M_star, alignment):
    # Block Diagonalization Loss
    S = torch.abs(M_star)
    STS = torch.matmul(S.transpose(-2, -1), S)
    if alignment:
        laploss_sts = tracenorm_of_normalized_laplacian(
            torch.mean(STS, 0))
    else:
        laploss_sts = torch.mean(
            tracenorm_of_normalized_laplacian(STS), 0)
    return laploss_sts


def loss_orth(M_star):
    # Orthogonalization of M
    I = make_identity_like(M_star)
    return torch.mean(torch.sum((I-M_star @ M_star.transpose(-2, -1))**2, axis=(-2, -1)))


class LinearTensorDynamicsLSTSQ(nn.Module):

    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.M = M

        def __call__(self, H):
            return H @ _rep_M(self.M, T=H.shape[1])

        def inverse(self, H):
            M = _rep_M(self.M, T=H.shape[1])
            return torch.linalg.solve(M, H.transpose(-2, -1)).transpose(-2, -1)

    def __init__(self, alignment=True):
        super().__init__()
        self.alignment = alignment

    def __call__(self, H, return_loss=False, fix_indices=None):
        # Regress M.
        # Note: backpropagation is disabled when fix_indices is not None.

        # H0.shape = H1.shape [n, t, s, a]
        H0, H1 = H[:, :-1], H[:, 1:]
        # num_ts x ([len_ts -1] * dim_s) x dim_a
        # The difference between the the time shifted components
        loss_internal_0 = _loss(H0, H1)
        ppe.reporting.report({
            'loss_internal_0': loss_internal_0.item()
        })
        _H0 = H0.reshape(H0.shape[0], -1, H0.shape[-1])
        _H1 = H1.reshape(H1.shape[0], -1, H1.shape[-1])
        if fix_indices is not None:
            # Note: backpropagation is disabled.
            dim_a = _H0.shape[-1]
            active_indices = np.array(list(set(np.arange(dim_a)) - set(fix_indices)))
            _M_star = _solve(_H0[:, :, active_indices],
                             _H1[:, :, active_indices])
            M_star = make_identity(_H1.shape[0], _H1.shape[-1], _H1.device)
            M_star[:, active_indices[:, np.newaxis], active_indices] = _M_star
        else:
            M_star = _solve(_H0, _H1)
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(dyn_fn(H0), H1)
        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })

        # M_star is returned in the form of module, not the matrix
        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, losses
        else:
            return dyn_fn


class HigherOrderLinearTensorDynamicsLSTSQ(LinearTensorDynamicsLSTSQ):

    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.M = M

        def __call__(self, Hs):
            nHs = [None]*len(Hs)
            for l in range(len(Hs)-1, -1, -1):
                if l == len(Hs)-1:
                    nHs[l] = Hs[l] @ _rep_M(self.M, Hs[l].shape[1])
                else:
                    nHs[l] = Hs[l] @ nHs[l+1]
            return nHs

    def __init__(self, alignment=True, n_order=2):
        super().__init__(alignment)
        self.n_order = n_order

    def __call__(self, H, return_loss=False, fix_indices=None):
        assert H.shape[1] > self.n_order
        # H0.shape = H1.shape [n, t, s, a]
        H0, Hn = H[:, :-self.n_order], H[:, self.n_order:]
        loss_internal_0 = _loss(H0, Hn)
        ppe.reporting.report({
            'loss_internal_0': loss_internal_0.item()
        })
        Ms = []
        _H = H
        if fix_indices is not None:
            raise NotImplementedError
        else:
            for n in range(self.n_order):
                # H0.shape = H1.shape [n, t, s, a]
                _H0, _H1 = _H[:, :-1], _H[:, 1:]
                if n == self.n_order - 1:
                    _H0 = _H0.reshape(_H0.shape[0], -1, _H0.shape[-1])
                    _H1 = _H1.reshape(_H1.shape[0], -1, _H1.shape[-1])
                    _H = _solve(_H0, _H1)  # [N, a, a]
                else:
                    _H = _solve(_H0, _H1)[:, 1:]  # [N, T-n, a, a]
                Ms.append(_H)
        dyn_fn = self.DynFn(Ms[-1])
        loss_internal_T = _loss(dyn_fn([H0] + Ms[:-1])[0], Hn)
        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })
        # M_star is returned in the form of module, not the matrix
        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, Ms[:-1], losses
        else:
            return dyn_fn, Ms[:-1]

# The fixed block model


class MultiLinearTensorDynamicsLSTSQ(LinearTensorDynamicsLSTSQ):

    def __init__(self, dim_a, alignment=True, K=4):
        super().__init__(alignment=alignment)
        self.dim_a = dim_a
        self.alignment = alignment
        assert dim_a % K == 0
        self.K = K

    def __call__(self, H, return_loss=False, fix_indices=None):
        H0, H1 = H[:, :-1], H[:, 1:]
        # num_ts x ([len_ts -1] * dim_s) x dim_a

        # The difference between the the time shifted components
        loss_internal_0 = _loss(H0, H1)

        _H0 = H0.reshape(H.shape[0], -1, H.shape[3])
        _H1 = H1.reshape(H.shape[0], -1, H.shape[3])

        ppe.reporting.report({
            'loss_internal_0': loss_internal_0.item()
        })
        M_stars = []
        for k in range(self.K):
            if fix_indices is not None and k in fix_indices:
                M_stars.append(make_identity(
                    H.shape[0], self.dim_a//self.K, H.device))
            else:
                st = k*(self.dim_a//self.K)
                ed = (k+1)*(self.dim_a//self.K)
                M_stars.append(_solve(_H0[:, :, st:ed], _H1[:, :, st:ed]))

        # Contstruct block diagonals
        for k in range(self.K):
            if k == 0:
                M_star = M_stars[0]
            else:
                M1 = M_star
                M2 = M_stars[k]
                _M1 = torch.cat(
                    [M1, torch.zeros(H.shape[0], M2.shape[1], M1.shape[2]).to(H.device)], axis=1)
                _M2 = torch.cat(
                    [torch.zeros(H.shape[0], M1.shape[1], M2.shape[2]).to(H.device), M2], axis=1)
                M_star = torch.cat([_M1, _M2], axis=2)
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(dyn_fn(H0), H1)
        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })

        # M_star is returned in the form of module, not the matrix
        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, losses
        else:
            return dyn_fn
