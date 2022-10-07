import torch
import torch.nn as nn
from einops import repeat
from utils.laplacian import tracenorm_of_normalized_laplacian, make_identity_like


def optimize_bd_cob(mats, batchsize=32, n_epochs=50, epochs_monitor=10):
    # Optimize change of basis matrix U by minimizing block diagonalization loss

    class ChangeOfBasis(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.U = nn.Parameter(torch.empty(d, d))
            torch.nn.init.orthogonal_(self.U)

        def __call__(self, mat):
            _U = repeat(self.U, "a1 a2 -> n a1 a2", n=mat.shape[0])
            n_mat = torch.linalg.solve(_U, mat) @ _U
            return n_mat

    change_of_basis = ChangeOfBasis(mats.shape[-1]).to(mats.device)
    dataloader = torch.utils.data.DataLoader(
        mats, batch_size=batchsize, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(change_of_basis.parameters(), lr=0.1)
    for ep in range(n_epochs):
        total_loss, total_N = 0, 0
        for mat in dataloader:
            n_mat = change_of_basis(mat)
            n_mat = torch.abs(n_mat)
            n_mat = torch.matmul(n_mat.transpose(-2, -1), n_mat)
            loss = torch.mean(
                tracenorm_of_normalized_laplacian(n_mat))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mat.shape[0]
            total_N += mat.shape[0]
        if ((ep+1) % epochs_monitor) == 0:
            print('ep:{} loss:{}'.format(ep, total_loss/total_N))
    return change_of_basis
