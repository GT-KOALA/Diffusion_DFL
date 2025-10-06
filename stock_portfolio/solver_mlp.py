import torch
import cvxpy as cp
import numpy as np
import torch.nn.functional as F
import time
from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt, dg_dz, d2g_dzdy
from model import MLP, GaussianMLP
import wandb

def compute_score(mu, logvar, y):
    ll_vec = -0.5 * (logvar + (y - mu)**2 * torch.exp(-logvar))  # (B, n)
    score = ll_vec.sum(dim=1)  
    return score

class MLPCvxpyModule(torch.nn.Module):
    def __init__(self, MLP_model, params, mc_samples):
        super().__init__()
        self.MLP_model = MLP_model
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_portfolio_parallel_kkt(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.alpha = params["alpha"]
        self.eps = 1e-12

    def _sample(self, x):
        if self.mc_samples > 1:
            x = x.repeat(self.mc_samples, 1) # (batch_size * mc_samples, 24)

        if isinstance(self.MLP_model, GaussianMLP):
            mu, logvar = self.MLP_model(x)   # (batch_size * mc_samples, 24) if mc_samples > 1 else (batch_size, 24)
            sigma = torch.exp(0.5 * logvar)
            y_preds = mu + torch.randn_like(mu) * sigma
        elif isinstance(self.MLP_model, MLP):
            y_preds = self.MLP_model(x)   # (batch_size * mc_samples, 24)
        # print(f"y_preds.max = {y_preds.abs().max()}, y_preds.min = {y_preds.abs().min()}")
        # grad_y_preds = self.diffusion.compute_elbo_grad(y_preds, x)
        return y_preds
    
    def forward(self, x):
        y_preds = self._sample(x)
        if self.mc_samples > 1:
            y_preds = y_preds.view(self.mc_samples, -1, self.n) # (mc_samples, batch_size, 24)
            y_preds = y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples, 24)

        with torch.no_grad():
            if self.mc_samples > 1:
                z_star_tuple, info = self.cvxpy_layer(y_preds)
                z_star_tensor = z_star_tuple[0]
            else:
                z_star_tuple, info = self.cvxpy_layer(y_preds.unsqueeze(1))
                z_star_tensor = z_star_tuple[0]

        MLP_model = self.MLP_model
        device = y_preds.device
    
        class MLPCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star):
                ctx.save_for_backward(y_preds, z_star)
                return y_preds, z_star
            
            @staticmethod
            def backward(ctx, dl_dy_preds, dl_dz_star):
                if dl_dz_star.norm() == 0: # loss function only depends on y_preds (e.g. MSE)
                    return dl_dy_preds, None
                y_preds, z_star = ctx.saved_tensors
                kkt = torch.from_numpy(info["KKT_matrices"]).float().to(x.device)

                B, m = kkt.shape[0], kkt.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))
                z_star_unsq = z_star.unsqueeze(1) # (batch_size, 1, 24)

                # d2f_dzy = 2 * self.alpha * y_preds * z_star_unsq
                # d2f_dzy = torch.diag_embed(d2f_dzy)
                _d2g_dzdy = d2g_dzdy(y_preds, z_star, self.params)
                d2f_dzy_padded = F.pad(_d2g_dzdy, (0, 0, 0, m - n))

                # kkt_exp = kkt.unsqueeze(1).repeat(1, self.mc_samples, 1, 1) # (batch_size, mc_samples, 24, 24)
                # k_inv_dg = torch.linalg.solve(kkt_exp, d2f_dzy_padded) # (batch_size, mc_samples, 24, 24)
                k_inv_dg = torch.linalg.solve(kkt, d2f_dzy_padded)
                
                # for i in range(self.mc_samples):        
                #     # Point estimation
                #     dl_dy = - torch.einsum('bi,bsij->bsj', dl_dz_star_padded, k_inv_dg) # (batch_size, mc_samples, 24)
                
                # dl_dy = dl_dy / float(self.mc_samples)

                # return dl_dy, None
                weighted_grad_y_preds_list = []
                for i in range(self.mc_samples):        
                    # Point estimation
                    dl_dy = - torch.einsum('bi,bij->bj', dl_dz_star_padded, k_inv_dg)
                    weighted_grad_y_preds_list.append(dl_dy)

                if self.mc_samples > 1:
                    weighted_grad_y_preds = torch.stack(weighted_grad_y_preds_list, dim=1) # (batch_size, mc_samples, 24)
                    weighted_grad_y_preds = weighted_grad_y_preds / float(self.mc_samples)
                else:
                    weighted_grad_y_preds = weighted_grad_y_preds_list[0] # (batch_size, 24)

                return weighted_grad_y_preds, None

        return MLPCvxpyFn.apply(y_preds, z_star_tensor)
    
class MLPCvxpyModule_distr(torch.nn.Module):
    def __init__(self, MLP_model, params, mc_samples):
        super().__init__()
        self.MLP_model = MLP_model
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_portfolio_parallel_kkt(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.total_samples = mc_samples

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.total_samples, 1) # (batch_size * mc_samples * 5, 24)

        if isinstance(self.MLP_model, GaussianMLP):
            mu, logvar = self.MLP_model(x)   # (batch_size * mc_samples, 24)
            sigma = torch.exp(0.5 * logvar)
            with torch.no_grad():
                y_preds = mu + torch.randn_like(mu) * sigma
        else:
            print("Have to be GaussianMLP")
            exit()

        time_start = time.time()
        elbo_losses = compute_score(mu, logvar, y_preds)
        elbo_losses = elbo_losses.view(self.total_samples, -1) # (mc_samples * 5, batch_size)
        time_end = time.time()
        # print(f"Time for computing elbo = {time_end - time_start}")

        return y_preds, elbo_losses
    
    def forward(self, x, test_mode=True):
        _y_preds, _elbo_losses = self._sample(x, test_mode=test_mode)

        if self.total_samples > 1:
            _y_preds = _y_preds.view(self.total_samples, -1, self.n) # (mc_samples * 5, batch_size, 24)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples * 5, 24)

        if self.mc_samples > 1:
            _y_preds_input = _y_preds[:, :self.mc_samples, :]
        else:
            _y_preds_input = _y_preds[:, 0, :].squeeze(1)

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
        time_end = time.time()
        # print(f"Time for cvxpy optimization = {time_end - time_start}")
        z_star_tensor = z_star_tuple[0]

        device = _y_preds_input.device

        _MLP_params = tuple(p for p in self.MLP_model.parameters() if p.requires_grad)

        class MLPCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, *MLP_params):
                ctx.save_for_backward(y_preds, z_star, *MLP_params)
                return y_preds, z_star
            
            @staticmethod
            def backward(ctx, dl_dy_preds, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, *MLP_params = ctx.saved_tensors
                kkt = torch.from_numpy(info["KKT_matrices"]).to(x.device)

                B, m = kkt.shape[0], kkt.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                if y_preds.ndim == 3:
                    y_preds_mc = y_preds.permute(1, 0, 2).contiguous() # (S * 5, B, n)
                else:
                    y_preds_mc = y_preds.unsqueeze(0) # (1, B, n)

                # _weight_losses = []
                # for i in range(self.total_samples):
                #     dg_dz_i = dg_dz(y_preds_mc[i], z_star, self.gs, self.ge) # (batch_size, n)
                #     dg_dz_i_padded = F.pad(dg_dz_i, (0, m - n)) # (batch_size, m)
                #     w_i = comp_w_i(dl_dz_star_padded, inv_matrices, dg_dz_i_padded) # (batch_size, )
                #     _weight_losses.append(w_i * _elbo_losses[i])
                # _weight_losses = torch.stack(_weight_losses, dim=0) # (S * 5, B)
                
                # Same w * _elbo_losses as above
                dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples, 1), self.params)          # (S * 5 * B, n)
                dg_dz_all = dg_dz_flat.contiguous().view(self.total_samples, B, n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)
                dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).repeat(self.total_samples, 1, 1)

                kkt_exp = kkt.unsqueeze(0).repeat(self.total_samples, 1, 1, 1)    # (S, B, m, m)
                rhs = dg_dz_pad.unsqueeze(-1)                                  # (S, B, m, 1)
                k_inv_dg = torch.linalg.solve(kkt_exp, rhs).squeeze(-1)           # (S, B, m)

                w = - (dl_dz_star_pad_exp * k_inv_dg).sum(dim=-1)   

                with torch.enable_grad():
                    _elbo_losses_clamped = _elbo_losses.clamp(max=0.0)
                    weighted_elbo = (w.detach() * _elbo_losses_clamped).sum(dim=1).mean() / float(self.total_samples)

                weighted_elbo_grads = torch.autograd.grad(
                    weighted_elbo,
                    MLP_params,
                    retain_graph=True,
                    allow_unused=False,
                    create_graph=False
                )
                time_end = time.time()
                # print(f"Time for backward = {time_end - time_start}")

                return None, None, *tuple(weighted_elbo_grads)

        return MLPCvxpyFn.apply(_y_preds, z_star_tensor, *_MLP_params)