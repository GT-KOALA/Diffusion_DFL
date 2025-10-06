import torch
import cvxpy as cp
import numpy as np
import torch.nn.functional as F
import time
from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt, dg_dz

class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion
        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_portfolio_parallel_kkt(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.total_samples = mc_samples
        self.resample = resample

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.total_samples, 1) # (batch_size * mc_samples * 5, 24)
        time_start = time.time()
        with torch.no_grad():
            y_preds = self.diffusion.sample_elbo(x, test_mode=True)   # no need for gradient (batch_size * mc_samples, 24)
        time_end = time.time()
        # print(f"Time for sampling = {time_end - time_start}")

        # del x
        # torch.cuda.empty_cache()
        # grad_y_preds = self.pretrain_diffusion.compute_elbo_grad(y_preds, x)
        # print(f"y_preds.max = {y_preds.max()}, y_preds.min = {y_preds.min()}, y_preds.mean = {y_preds.abs().mean()}")
        return y_preds
    
    def forward(self, _x, test_mode=False):
        _y_preds = self._sample(_x, test_mode=test_mode)
        ori_y_preds = _y_preds.clone()

        if self.total_samples > 1:
            _y_preds = _y_preds.contiguous().view(self.total_samples, -1, self.n) # (mc_samples * 5, batch_size, 24)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples * 5, 24)
        else:
            _y_preds = _y_preds.unsqueeze(0).permute(1, 0, 2).contiguous() # (64, 1, 24)

        if self.mc_samples > 1:
            _y_preds_input = _y_preds[:, :self.mc_samples, :]
        else:
            _y_preds_input = _y_preds

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
        time_end = time.time()
        # print(f"Time for cvxpy optimization = {time_end - time_start}")
        z_star_tensor = z_star_tuple[0]

        diffusion = self.diffusion
        device = diffusion.device
        resample = self.resample

        _diffusion_params = tuple(p for p in diffusion.model_net.parameters() if p.requires_grad)

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, x, *diffusion_params):
                ctx.save_for_backward(y_preds, z_star, x, *diffusion_params)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, x, *diffusion_params = ctx.saved_tensors
                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]
                # inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)
                kkt = torch.from_numpy(info["KKT_matrices"]).to(y_preds.device)

                B, m = kkt.shape[0], kkt.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                if y_preds.ndim == 3:
                    y_preds_mc = y_preds.permute(1, 0, 2).contiguous() # (S * 5, B, n)
                else:
                    y_preds_mc = y_preds.unsqueeze(0) # (1, B, n)

                dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples, 1), self.params)          # (S * 5 * B, n)
                dg_dz_all = dg_dz_flat.contiguous().view(self.total_samples, B, n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)
                dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).repeat(self.total_samples, 1, 1)

                kkt_exp = kkt.unsqueeze(0).repeat(self.total_samples, 1, 1, 1)    # (S, B, m, m)
                rhs = dg_dz_pad.unsqueeze(-1)                                     # (S, B, m, 1)
                k_inv_dg = torch.linalg.solve(kkt_exp, rhs).squeeze(-1)           # (S, B, m)

                w = - (dl_dz_star_pad_exp * k_inv_dg).sum(dim=-1)   
                _y_preds_sample, _x_sample, _w_sample = ori_y_preds, x.repeat(self.total_samples, 1), w.detach()

                with torch.enable_grad():
                    _elbo_losses = self.diffusion.compute_elbo_grad_vec(_y_preds_sample, _x_sample, resample=resample)
                    _elbo_losses = _elbo_losses.contiguous().view(self.total_samples, -1)
                    weighted_elbo = (_w_sample.detach() * _elbo_losses).sum() / self.total_samples
                weighted_elbo_grads = torch.autograd.grad(
                    weighted_elbo,
                    diffusion_params,
                    retain_graph=False,
                    allow_unused=False,
                    create_graph=False
                )

                return None, None, None, *tuple(weighted_elbo_grads)

        return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor, _x, *_diffusion_params)