import torch
import cvxpy as cp
import numpy as np
import torch.nn.functional as F
import time
from cvxpy_stock_portfolio import cvxpy_portfolio, dg_dz

class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True):
        super().__init__()
        self.diffusion = diffusion
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_portfolio(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.total_samples = mc_samples

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.total_samples, 1) # (batch_size * mc_samples * 5, 24)
        time_start = time.time()
        with torch.no_grad():
            y_preds = self.diffusion.sample_elbo(x, test_mode=True)   # no need for gradient (batch_size * mc_samples, 24)
        time_end = time.time()
        # print(f"Time for sampling = {time_end - time_start}")

        time_start = time.time()
        elbo_losses = self.diffusion.compute_elbo_grad_vec(y_preds, x)
        elbo_losses = elbo_losses.contiguous().view(self.total_samples, -1) # (mc_samples * 5, batch_size)
        time_end = time.time()
        # print(f"Time for computing elbo = {time_end - time_start}")

        del x
        torch.cuda.empty_cache()
        # grad_y_preds = self.pretrain_diffusion.compute_elbo_grad(y_preds, x)
        return y_preds, elbo_losses
    
    def forward(self, x, test_mode=False):
        _y_preds, _elbo_losses = self._sample(x, test_mode=test_mode)

        if self.total_samples > 1:
            _y_preds = _y_preds.contiguous().view(self.total_samples, -1, self.n) # (mc_samples * 5, batch_size, 24)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples * 5, 24)

        # print(f"_y_preds.max = {_y_preds.abs().max()}")

        if self.mc_samples > 1:
            _y_preds_input = _y_preds[:, :self.mc_samples, :].contiguous()
        else:
            _y_preds_input = _y_preds[:, 0, :].squeeze(1)

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
        time_end = time.time()
        # print(f"Time for cvxpy optimization = {time_end - time_start}")
        z_star_tensor = z_star_tuple[0]

        diffusion = self.diffusion
        device = diffusion.device

        _diffusion_params = tuple(p for p in diffusion.model_net.parameters() if p.requires_grad)

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, _elbo_losses, *diffusion_params):
                ctx.save_for_backward(y_preds, z_star, _elbo_losses, *diffusion_params)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, _elbo_losses, *diffusion_params = ctx.saved_tensors
                inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)

                B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                if y_preds.ndim == 3:
                    y_preds_mc = y_preds.permute(1, 0, 2).contiguous() # (S * 5, B, n)
                else:
                    y_preds_mc = y_preds.unsqueeze(0) # (1, B, n)

                dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples, 1), self.params)          # (S * 5 * B, n)
                dg_dz_all = dg_dz_flat.view(self.total_samples, B, n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)

                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]

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
                        diffusion_params,
                        retain_graph=True,
                        allow_unused=False,
                        create_graph=False
                    )
                del _elbo_losses
                torch.cuda.empty_cache()

                time_end = time.time()
                # print(f"Time for backward = {time_end - time_start}")

                return None, None, None, *tuple(weighted_elbo_grads)

        return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor, _elbo_losses, *_diffusion_params)