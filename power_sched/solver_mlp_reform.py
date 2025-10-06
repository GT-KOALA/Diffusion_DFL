import torch
import cvxpy as cp
import numpy as np
import torch.nn.functional as F
import time
from cvxpy_powers_sched import cvxpy_ps, cvxpy_layer, dg_dz
from model_classes import MLP, GaussianMLP, SolveScheduling

def compute_score(mu, logvar, y):
    ll_vec = -0.5 * (logvar + (y - mu)**2 * torch.exp(-logvar))  # (B, n)
    score = ll_vec.sum(dim=1)  
    return score

class MLPCvxpyModule(torch.nn.Module):
    def __init__(self, MLP_model, params, mc_samples, distr_est=True):
        super().__init__()
        self.MLP_model = MLP_model
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.cvxpy_layer = SolveScheduling(params)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est

    def _sample(self, x):
        x = x.repeat(self.mc_samples, 1) # (batch_size * mc_samples, 24)
        mu, logvar = self.MLP_model(x)   # (batch_size * mc_samples, 24)
            
        return mu, logvar
    
    def forward(self, x):
        mu, logvar = self._sample(x)
        if self.mc_samples > 1:
            mu = mu.view(self.mc_samples, -1, self.n) # (mc_samples, batch_size, 24)
            logvar = logvar.view(self.mc_samples, -1, self.n) # (mc_samples, batch_size, 24)
            mu = mu.permute(1, 0, 2).contiguous() # (batch_size, mc_samples, 24)
            logvar = logvar.permute(1, 0, 2).contiguous() # (batch_size, mc_samples, 24)

        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(mu, logvar)
            z_star_tensor = z_star_tuple[0]

        MLP_model = self.MLP_model
        device = mu.device
        distr_est = self.distr_est
    
        class MLPCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, mu, logvar, z_star):
                ctx.save_for_backward(mu, logvar, z_star)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                mu, logvar, z_star = ctx.saved_tensors
                inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)

                B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                dg_dy = - torch.eye(n, device=device)
                dg_dy = dg_dy.unsqueeze(0).expand(B, -1, -1) # (batch_size, n, n)
                dg_dy_padded = F.pad(dg_dy, (0, 0, 0, m - n))

                weighted_grad_y_preds_list = []
                for i in range(self.mc_samples):        
                    # Point estimation
                    dl_dy = - torch.einsum('bi,bij,bjk->bk',
                                dl_dz_star_padded,
                                inv_matrices,
                                dg_dy_padded)
                    weighted_grad_y_preds_list.append(dl_dy)

                if self.mc_samples > 1:
                    weighted_grad_y_preds = torch.stack(weighted_grad_y_preds_list, dim=1) # (batch_size, mc_samples, 24)
                    weighted_grad_y_preds = weighted_grad_y_preds / float(self.mc_samples)
                else:
                    weighted_grad_y_preds = weighted_grad_y_preds_list[0]

                # import pdb; pdb.set_trace()

                return weighted_grad_y_preds, None

        return MLPCvxpyFn.apply(mu, logvar, z_star_tensor)
    
class MLPCvxpyModule_distr(torch.nn.Module):
    def __init__(self, MLP_model, params, mc_samples, distr_est=True):
        super().__init__()
        self.MLP_model = MLP_model
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_ps(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.total_samples = mc_samples

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.total_samples, 1) # (batch_size * mc_samples * 5, 24)

        if isinstance(self.MLP_model, GaussianMLP):
            mu, logvar = self.MLP_model(x)   # (batch_size * mc_samples, 24)
            sigma = torch.exp(0.5 * logvar)
            y_preds = mu + torch.randn_like(mu) * sigma
        else:
            print("Have to be GaussianMLP")
            exit()

        time_start = time.time()
        elbo_losses = compute_score(mu, logvar, y_preds)
        elbo_losses = elbo_losses.view(self.total_samples, -1) # (mc_samples * 5, batch_size)
        time_end = time.time()
        print(f"Time for computing elbo = {time_end - time_start}")

        del x
        torch.cuda.empty_cache()
        # grad_y_preds = self.pretrain_diffusion.compute_elbo_grad(y_preds, x)
        return y_preds, elbo_losses
    
    def forward(self, x, test_mode=False):
        _y_preds, _elbo_losses = self._sample(x, test_mode=test_mode)

        if self.total_samples > 1:
            _y_preds = _y_preds.view(self.total_samples, -1, self.n) # (mc_samples * 5, batch_size, 24)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples * 5, 24)

        # print(f"_y_preds.max = {_y_preds.abs().max()}")

        if self.mc_samples > 1:
            _y_preds_input = _y_preds[:, :self.mc_samples, :]
        else:
            _y_preds_input = _y_preds[:, 0, :].squeeze(1)

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
        time_end = time.time()
        print(f"Time for cvxpy optimization = {time_end - time_start}")
        z_star_tensor = z_star_tuple[0]

        device = _y_preds_input.device
        distr_est = self.distr_est

        _MLP_params = tuple(p for p in self.MLP_model.parameters() if p.requires_grad)

        class MLPCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, *MLP_params):
                ctx.save_for_backward(y_preds, z_star, *MLP_params)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, *MLP_params = ctx.saved_tensors
                inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)

                B, m = inv_matrices.shape[0], inv_matrices.shape[1]
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
                dg_dz_all = dg_dz_flat.view(self.total_samples, B, n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)

                weighted_elbo_grads = [torch.zeros_like(p) for p in MLP_params]

                if distr_est:
                    dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).expand(self.total_samples, -1, -1)   # (S * 5,B,m)
                    inv_matrices_exp   = inv_matrices.unsqueeze(0).expand(self.total_samples, -1, -1, -1)    # (S * 5,B,m,m)
                    w = - torch.einsum('sbi,sbij,sbj->sb',
                            dl_dz_star_pad_exp,
                            inv_matrices_exp,
                            dg_dz_pad) # (S * 5, B)

                    with torch.enable_grad():
                        weighted_elbo = (w.detach() * _elbo_losses).sum(dim=1).mean() / float(self.total_samples)
                        weighted_elbo_grads = torch.autograd.grad(
                            weighted_elbo,
                            MLP_params,
                            retain_graph=True,
                            allow_unused=False,
                            create_graph=False
                        )
                time_end = time.time()
                print(f"Time for backward = {time_end - time_start}")

                return None, None, *tuple(weighted_elbo_grads)

        return MLPCvxpyFn.apply(_y_preds, z_star_tensor, *_MLP_params)