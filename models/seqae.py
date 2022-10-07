import numpy as np
import torch
import torch.nn as nn
from models import dynamics_models
import torch.nn.utils.parametrize as P
from models.dynamics_models import LinearTensorDynamicsLSTSQ, MultiLinearTensorDynamicsLSTSQ, HigherOrderLinearTensorDynamicsLSTSQ
from models.base_networks import ResNetEncoder, ResNetDecoder, Conv1d1x1Encoder
from einops import rearrange, repeat
from utils.clr import simclr


class SeqAELSTSQ(nn.Module):
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            *args,
            **kwargs):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = LinearTensorDynamicsLSTSQ(alignment=alignment)
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def _encode_base(self, xs, enc):
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)
        H = torch.reshape(
            H, (shape[0], shape[1], *H.shape[1:]))
        return H

    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        H = torch.reshape(
            H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis,
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        return H

    def phi(self, xs):
        return self._encode_base(xs, self.enc.phi)

    def get_M(self, xs):
        dyn_fn = self.dynamics_fn(xs)
        return dyn_fn.M

    def decode(self, H):
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis),
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]
        if hasattr(self, "pidec"):
            H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
            H = self.pidec(H)
        else:
            H = rearrange(H, 'n t d_s d_a -> (n t) (d_s d_a)')
        x_next_preds = self.dec(H)
        x_next_preds = torch.reshape(
            x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds

    def dynamics_fn(self, xs, return_loss=False, fix_indices=None):
        H = self.encode(xs)
        return self.dynamics_model(H, return_loss=return_loss, fix_indices=fix_indices)

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond, predictive=self.predictive, reconst=reconst)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
        if reconst:
            xs_target = xs
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]
        loss = torch.mean(
            torch.sum((xs_target - xs_pred) ** 2, axis=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss

    def __call__(self, xs_cond, return_reg_loss=False, n_rolls=1, fix_indices=None, predictive=True, reconst=False):
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a
        H = self.encode(xs_cond)

        # ==Esitmate dynamics==
        ret = self.dynamics_model(
            H, return_loss=return_reg_loss, fix_indices=fix_indices)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, losses = ret
        else:
            fn = ret

        if predictive:
            H_last = H[:, -1:]
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        x_preds = self.decode(H_preds)
        if return_reg_loss:
            return x_preds, losses
        else:
            return x_preds
        
        
    def loss_equiv(self, xs, T_cond=2, reduce=False):
        bsize = len(xs)
        xs_cond = xs[:, :T_cond]
        xs_target = xs[:, T_cond:]
        H = self.encode(xs_cond[:, -1:])
        dyn_fn = self.dynamics_fn(xs_cond)
        
        H_last = H
        H_preds = []
        n_rolls = xs.shape[1] - T_cond
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds.append(H_last)
        H_pred = torch.cat(H_preds, axis=1)
        # swapping M
        dyn_fn.M = dyn_fn.M[torch.arange(-1, bsize-1)]
        
        H_last = H
        H_preds_perm = []
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds_perm.append(H_last)
        H_pred_perm = torch.cat(H_preds_perm, axis=1)
        
        xs_pred = self.decode(H_pred)
        xs_pred_perm = self.decode(H_pred_perm)
        reduce_dim = (1,2,3,4,5) if reduce else (2,3,4)
        loss = torch.sum((xs_target-xs_pred)**2, dim=reduce_dim).detach().cpu().numpy()
        loss_perm = torch.sum((xs_target-xs_pred_perm)**2, dim=reduce_dim).detach().cpu().numpy()
        return loss, loss_perm
        

class SeqAEHOLSTSQ(SeqAELSTSQ):
    # Higher order version of SeqAELSTSQ
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            n_order=2,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = HigherOrderLinearTensorDynamicsLSTSQ(
            alignment=alignment, n_order=n_order)
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        if reconst:
            raise NotImplementedError
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, predictive=self.predictive, return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
        xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]
        loss = torch.mean(
                torch.sum((xs_target - xs_pred) ** 2, axis=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss


    def __call__(self, xs, n_rolls=1, fix_indices=None, predictive=True, return_reg_loss=False):
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a
        H = self.encode(xs)

        # ==Esitmate dynamics==
        ret = self.dynamics_model(
            H, return_loss=return_reg_loss, fix_indices=fix_indices)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, Ms, losses = ret
        else:
            fn, Ms = ret

        if predictive:
            Hs_last = [H[:, -1:]] + [M[:, -1:] for M in Ms]
            array = np.arange(n_rolls)
        else:
            Hs_last = [H[:, :1]] + [M[:, :1] for M in Ms]
            array = np.arange(xs.shape[1] + n_rolls - 1)

        # Create prediction for the unseen future
        H_preds = []
        for _ in array:
            Hs_last = fn(Hs_last)
            H_preds.append(Hs_last[0])
        H_preds = torch.cat(H_preds, axis=1)
        x_preds = self.decode(H_preds)
        if return_reg_loss:
            return x_preds, losses
        else:
            return x_preds

    def loss_equiv(self, xs, T_cond=5, reduce=False, return_generated_images=False):
        bsize = len(xs)
        xs_cond = xs[:, :T_cond]
        xs_target = xs[:, T_cond:]
        H = self.encode(xs_cond[:, -1:])
        dyn_fn, Ms = self.dynamics_fn(xs_cond)

        H_last = [H] + [M[:, -1:] for M in Ms]
        H_preds = []
        n_rolls = xs.shape[1] - T_cond
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds.append(H_last[0])
        H_pred = torch.cat(H_preds, axis=1)
        # swapping M
        dyn_fn.M = dyn_fn.M[torch.arange(-1, bsize-1)]
        Ms = [M[torch.arange(-1, bsize-1)] for M in Ms]
        
        H_last = [H] + [M[:, -1:] for M in Ms]
        H_preds_perm = []
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds_perm.append(H_last[0])
        H_pred_perm = torch.cat(H_preds_perm, axis=1)
        
        xs_pred = self.decode(H_pred)
        xs_pred_perm = self.decode(H_pred_perm)
        loss = torch.sum((xs_target-xs_pred)**2, dim=(2,3,4)).detach().cpu().numpy()
        loss_perm = torch.sum((xs_target-xs_pred_perm)**2, dim=(2,3,4)).detach().cpu().numpy()
        if reduce:
            loss = torch.mean(loss)
            loss_perm = torch.mean(loss_perm)
        if return_generated_images:
            return (loss, loss_perm), (xs_pred, xs_pred_perm)
        else:
            return loss, loss_perm



class SeqAEMultiLSTSQ(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            K=8,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.K = K
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = MultiLinearTensorDynamicsLSTSQ(
            dim_a, alignment=alignment, K=K)
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def get_blocks_of_M(self, xs):
        M = self.get_M(xs)
        blocks = []
        for k in range(self.K):
            dim_block = self.dim_a // self.K
            blocks.append(M[:, k*dim_block:(k+1)*dim_block]
                          [:, :, k*dim_block:(k+1)*dim_block])
        blocks_of_M = torch.stack(blocks, 1)
        return blocks_of_M


class SeqAENeuralM(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            ch_x=3,
            k=1.0,
            alignment=False,
            kernel_size=3,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.alignment = alignment
        self.initial_scale_M = 0.01
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.M_net = ResNetEncoder(
            dim_a*dim_a, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, n_blocks=n_blocks, bottom_width=bottom_width)

    def dynamics_fn(self, xs):
        M = self.get_M(xs)
        dyn_fn = dynamics_models.LinearTensorDynamicsLSTSQ.DynFn(M)
        return dyn_fn

    def get_M(self, xs):
        xs = rearrange(xs, 'n t c h w -> n (t c) h w')
        M = self.M_net(xs)
        M = rearrange(M, 'n (a_1 a_2) -> n a_1 a_2', a_1=self.dim_a)
        M = self.initial_scale_M * M
        return M

    def __call__(self, xs, n_rolls=1, return_reg_loss=False, predictive=True, reconst=False):
        # ==Esitmate dynamics==
        fn = self.dynamics_fn(xs)

        if reconst:
            H = self.encode(xs)
            if predictive:
                H_last = H[:, -1:]
            else:
                H_last = H[:, :1]
        else:
            H_last = self.encode(xs[:, -1:] if predictive else xs[:, :1])

        if predictive:
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)
        else:
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs.shape[1] + n_rolls - 1)

        # Create prediction for the unseen future
        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)
        x_preds = self.decode(H_preds)
        if return_reg_loss:
            losses = (dynamics_models.loss_bd(fn.M, self.alignment),
                      dynamics_models.loss_orth(fn.M), 0)
            return x_preds, losses
        else:
            return x_preds

class SeqAENeuralMLatentPredict(SeqAENeuralM):
    def __init__(self,
            dim_a,
            dim_m,
            ch_x=3,
            k=1.0,
            alignment=False,
            kernel_size=3,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            loss_latent_coeff=0,
            loss_pred_coeff=1.0,
            loss_reconst_coeff=0,
            normalize=True,
            *args,
            **kwargs):
        assert predictive
        super().__init__(
            dim_a=dim_a,
            dim_m=dim_m,
            ch_x=ch_x,
            k=k,
            alignment=alignment,
            kernel_size=kernel_size,
            predictive=predictive,
            bottom_width=bottom_width,
            n_blocks=n_blocks,
            )
        self.loss_reconst_coeff = loss_reconst_coeff
        self.loss_pred_coeff = loss_pred_coeff
        self.loss_latent_coeff = loss_latent_coeff
        self.normalize = normalize 

    def normalize_isotypic_copy(self, H):
        isotype_norm = torch.sqrt(torch.sum(H**2, axis=2, keepdims=True))
        H = H / isotype_norm
        return H

    #encoding  function with isotypic column normalization
    def encode(self, xs):
        H = super().encode(xs)
        if self.normalize:
            H = self.normalize_isotypic_copy(H)
        return H

    def latent_error(self, H_preds, H_target):
        latent_e = torch.mean(torch.sum((H_preds - H_target)**2, axis=(2,3)))
        return latent_e

    def obs_error(self, xs_1, xs_2):
        obs_e = torch.mean(torch.sum((xs_1 - xs_2)**2, axis=(2,3,4)))
        return obs_e

    def __call__(self, xs, n_rolls=1, T_cond=2, return_losses=False, return_reg_losses=False):
        xs_cond, xs_target =xs[:, :T_cond], xs[:, T_cond:]
        fn = self.dynamics_fn(xs_cond)
        H_cond, H_target = self.encode(xs_cond), self.encode(xs_target)
        H_last = H_cond[:, -1:]
        H_preds=[H_cond]
        array = np.arange(n_rolls)

        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)
        xs_preds = self.decode(H_preds)
        ret = [xs_preds]
        if return_losses:
            losses = {}
            losses['loss_reconst'] = self.obs_error(xs_preds[:, :T_cond], xs_cond) if self.loss_reconst_coeff > 0 else torch.tensor([0]).to(xs.device)
            losses['loss_pred'] = self.obs_error(xs_preds[:, T_cond:], xs_target) if self.loss_pred_coeff > 0 else torch.tensor([0]).to(xs.device)
            losses['loss_latent'] = self.latent_error(H_preds[:, T_cond:], H_target) if self.loss_latent_coeff > 0 else torch.tensor([0]).to(xs.device)
            ret += [losses]
        if return_reg_losses:
            ret += [(dynamics_models.loss_bd(fn.M, self.alignment),
                      dynamics_models.loss_orth(fn.M), 0)]
        return ret

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        ret = self(xs, return_losses=True, return_reg_losses=return_reg_loss, T_cond= T_cond,
                       n_rolls=xs.shape[1] - T_cond)
        if return_reg_loss:
            _, losses, reg_losses = ret
        else:
            _, losses = ret
            
        total_loss = self.loss_reconst_coeff * losses['loss_reconst'] \
            + self.loss_pred_coeff * losses['loss_pred'] \
            + self.loss_latent_coeff * losses['loss_latent'] 
        return (total_loss, reg_losses) if return_reg_loss else total_loss
   


class SeqAENeuralTransition(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            T_cond=2,
            bottom_width=4,
            n_blocks=3,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.T_cond = T_cond
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.ar = Conv1d1x1Encoder(dim_out=dim_a)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)

    def loss(self, xs, return_reg_loss=False, T_cond=2, reconst=False):
        assert T_cond == self.T_cond
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, n_rolls=xs.shape[1] - T_cond, reconst=reconst)
        xs_target = xs if reconst else xs[:, T_cond:]
        loss = torch.mean(
            torch.sum((xs_target - xs_pred) ** 2, axis=[2, 3, 4]))
        if return_reg_loss:
            return loss, [torch.Tensor(np.array(0, dtype=np.float32)).to(xs.device)] * 3
        else:
            return loss

    def get_M(self, xs):
        T = xs.shape[1]
        xs = rearrange(xs, 'n t c h w -> (n t) c h w')
        H = self.enc(xs)
        H = rearrange(H, '(n t) c -> n (t c)', t=T)
        return H

    def __call__(self, xs, n_rolls=1, reconst=False):
        # ==Esitmate dynamics==
        H = self.encode(xs)

        array = np.arange(n_rolls)
        H_preds = [H] if reconst else []
        # Create prediction for the unseen future
        for _ in array:
            H_pred = self.ar(rearrange(H, 'n t s a -> n (t a) s'))
            H_pred = rearrange(
                H_pred, 'n (t a) s-> n t s a', t=1, a=self.dim_a)
            H_preds.append(H_pred)
            H = torch.cat([H[:, 1:], H_pred], dim=1)
        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        return self.decode(H_preds)


class CPC(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            k=1.0,
            kernel_size=3,
            temp=0.01,
            normalize=True,
            loss_type='cossim',
            n_blocks=3,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.normalize = normalize
        self.temp = temp
        self.loss_type = loss_type
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.ar = Conv1d1x1Encoder(dim_out=dim_a*dim_m)

    def __call__(self, xs):
        H = self.encode(xs)  # [n, t, s, a]

        # Create prediction for the unseen future
        H = rearrange(H, 'n t s a -> n (t a) s')

        # Obtain c in CPC
        H_pred = self.ar(H)  # [n a s]
        H_pred = rearrange(H_pred, 'n a s -> n s a')

        return H_pred

    def get_M(self, xs):
        T = xs.shape[1]
        xs = rearrange(xs, 'n t c h w -> (n t) c h w')
        H = self.enc(xs)
        H = rearrange(H, '(n t) c -> n (t c)', t=T)
        return H

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        T_pred = xs.shape[1] - T_cond
        assert T_pred == 1
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a
        H = self.encode(xs)  # [n, t, s, a]

        # Create prediction for the unseen future
        H_cond = H[:, :T_cond]
        H_cond = rearrange(H_cond, 'n t s a -> n (t a) s')

        # Obtain c in CPC
        H_pred = self.ar(H_cond)  # [n a s]
        H_pred = rearrange(H_pred, 'n a s -> n s a')

        H_true = H[:, -1]  # n s a
        H_true = rearrange(H_true, 'n s a -> n (s a)')
        H_pred = rearrange(H_pred, 'n s a -> n (s a)')
        loss = simclr([H_pred, H_true], self.temp,
                      normalize=self.normalize, loss_type=self.loss_type)
        if return_reg_loss:
            reg_losses = [torch.Tensor(np.array(0, dtype=np.float32))]*3
            return loss, reg_losses
        else:
            return loss
