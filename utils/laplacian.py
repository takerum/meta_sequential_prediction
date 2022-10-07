import torch
import numpy as np
from einops import repeat


def make_identity(N, D, device):
    if N is None:
        return torch.Tensor(np.array(np.eye(D))).to(device)
    else:
        return torch.Tensor(np.array([np.eye(D)] * N)).to(device)

def make_identity_like(A):
    assert A.shape[-2] == A.shape[-1] # Ensure A is a batch of squared matrices
    device = A.device
    shape = A.shape[:-2]
    eye = torch.eye(A.shape[-1], device=device)[(None,)*len(shape)]
    return eye.repeat(*shape, 1, 1)


def make_diagonal(vecs):
    vecs = vecs[..., None].repeat(*([1,]*len(vecs.shape)), vecs.shape[-1])
    return vecs * make_identity_like(vecs)

# Calculate Normalized Laplacian
def tracenorm_of_normalized_laplacian(A):
    D_vec = torch.sum(A, axis=-1)
    D = make_diagonal(D_vec)
    L = D - A
    inv_A_diag = make_diagonal(
        1 / torch.sqrt(1e-10 + D_vec))
    L = torch.matmul(inv_A_diag, torch.matmul(L, inv_A_diag))
    sigmas = torch.linalg.svdvals(L)
    return torch.sum(sigmas, axis=-1)
