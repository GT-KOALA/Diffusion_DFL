import torch
import torch.nn.functional as F
import time
from cvxpy_toy_kkt import cvxpy_toy_parallel_kkt, dg_dz
import wandb


class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, num_iter=None, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion
        self.mc_samples = mc_samples
        self.total_samples = mc_samples
        self.cvxpy_layer = cvxpy_toy_parallel_kkt(params, self.mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.resample = resample
        self.num_iter = num_iter
        
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
    
    def forward(self, _x, _y=None, test_mode=False):
        if _y is not None:
            _y_preds = _y
        else:
            _y_preds = self._sample(_x, test_mode=test_mode)
        ori_y_preds = _y_preds.clone()

        if self.total_samples > 1:
            _y_preds = _y_preds.contiguous().view(self.total_samples, -1, self.n) # (mc_samples * 5, batch_size, 24)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples * 5, 24)
        else:
            _y_preds = _y_preds.unsqueeze(0).permute(1, 0, 2).contiguous() # (64, 1, 24)

        # print(f"_y_preds.max = {_y_preds.abs().max()}")

        if self.mc_samples > 1:
            _y_preds_input = _y_preds[:, :self.mc_samples, :]
        else:
            _y_preds_input = _y_preds

        time_start = time.time()
        with torch.no_grad():
            # z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
            cvxpy_layer = cvxpy_toy_parallel_kkt(self.params, _x.shape[0] * self.mc_samples)
            z_star_tuple, info = cvxpy_layer(_y_preds_input.reshape(1, -1, self.n))
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
                y_preds, z_star, x, *diffusion_params = ctx.saved_tensors
                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]
                # inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)
                kkt = torch.from_numpy(info["KKT_matrices"]).float().to(x.device)

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
                dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples * x.shape[0], 1), self.params)          # (S * 5 * B, n)
                dg_dz_all = dg_dz_flat.contiguous().view(self.total_samples, x.shape[0], n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)
                dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).repeat(self.total_samples, 1, 1)

                kkt_exp = kkt.unsqueeze(0).repeat(self.total_samples, 1, 1, 1)    # (S, B, m, m)
                rhs = dg_dz_pad.unsqueeze(-1)                                     # (S, B, m, 1)
                k_inv_dg = torch.linalg.solve(kkt_exp, rhs).squeeze(-1)           # (S, B, m)

                w = - (dl_dz_star_pad_exp * k_inv_dg).sum(dim=-1)   
                _y_preds_sample, _x_sample, _w_sample = ori_y_preds, x.repeat(self.total_samples, 1), w.detach()

                with torch.enable_grad():
                    num_iter = self.num_iter if self.num_iter is not None else 1000
                    _elbo_losses = self.diffusion.compute_elbo_grad_vec(_y_preds_sample, _x_sample, num_iter=num_iter, resample=self.resample)
                    _elbo_losses = _elbo_losses.contiguous().view(self.total_samples, -1)
                    wandb.log({"_elbo_losses_max": _elbo_losses.max(), "_elbo_losses_min": _elbo_losses.min()})

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