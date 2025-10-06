import torch
import cvxpy as cp
from cvxpylayer_local.cvxpylayer import CvxpyLayer
import qpth
# from qpth.qp import QPSolvers
# from qpth.solvers.pdipm import batch as pdipm_b
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.solvers.pdipm import batch as pdipm_b
from torch.autograd import Function
from util import bger, expandParam, extract_nBatch, ReplayBufferW
import numpy as np
import autograd.numpy as auto_np
from cvxpylayer_local import utils
import autograd 
from functorch import vmap
import torch.nn.functional as F
import time
from cvxpy_powers_sched import cvxpy_ps, cvxpy_layer, dg_dz, cvxpy_ps_parallel
import random
import wandb

class DiffusionCvxpyModuleReplay(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, interval, distr_est=True, buffer_size=5000):
        super().__init__()
        self.diffusion = diffusion
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_ps_parallel(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.total_samples = mc_samples
        self.replay_buffer = ReplayBufferW(capacity=buffer_size, device=diffusion.device)
        self.cnt = -1
        self.interval = interval
        self.old_elbo_losses = None

    def _sample(self, x, test_mode=False):
        batch_size = x.shape[0]
        x = x.repeat(self.total_samples, 1) # (batch_size * mc_samples * 5, 24)
        time_start = time.time()
        with torch.no_grad():
            y_preds = self.diffusion.sample_elbo(x, test_mode=True)   # no need for gradient (batch_size * mc_samples, 24)
        time_end = time.time()
        # print(f"Time for sampling = {time_end - time_start}")
        wandb.log({
            f"time/time_for_sampling_{self.total_samples}_{batch_size}" : time_end - time_start,
        })

        return y_preds
    
    def forward(self, _x, idx, epoch, test_mode=False):
        self.cnt += 1
        if epoch % self.interval == 0 or epoch == -1:
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
                z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
            time_end = time.time()
            # print(f"Time for cvxpy optimization = {time_end - time_start}")
            z_star_tensor = z_star_tuple[0]
        else:
            # _y_preds, _x_sampled, z_star_tensor_sampled, info = self.replay_buffer.sample_last()
            # batch_size = z_star_tensor_sampled.shape[0]
            # _x = _x_sampled[:batch_size, :]
            # ori_y_preds = _y_preds.clone()

            _y_preds_sample, _x_sample, _w_sample, _z_star_sample, _info_sample, _elbo_sample = self.replay_buffer.get_many(idx)

            # for debugging
            # with torch.no_grad():
            #     _y_preds_2 = self._sample(_x, test_mode=test_mode)
            #     _y_preds_2 = _y_preds_2.contiguous().view(self.total_samples, -1, self.n) # (mc_samples * 5, batch_size, 24)
            #     _y_preds_2 = _y_preds_2.permute(1, 0, 2).contiguous()
            #     _y_preds_2 = _y_preds_2[:, :self.mc_samples, :]
            #     _y_preds_2 = _y_preds_2.squeeze(1)
            #     z_star_tuple_2, info_2 = self.cvxpy_layer(_y_preds_2)
            #     z_star_tensor = z_star_tuple_2[0]

        diffusion = self.diffusion
        device = diffusion.device
        distr_est = self.distr_est

        _diffusion_params = tuple(p for p in diffusion.model_net.parameters() if p.requires_grad)

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, x, idx, *diffusion_params):
                ctx.save_for_backward(y_preds, z_star, x, idx, *diffusion_params)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, x, idx, *diffusion_params = ctx.saved_tensors
                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]

                assert distr_est
                if epoch % self.interval == 0 or epoch == -1: # -1 for the validation epoch
                    inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)
                    B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                    n = dl_dz_star.shape[1]      
                    dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))
                    if y_preds.ndim == 3:
                        y_preds_mc = y_preds.permute(1, 0, 2).contiguous() # (S * 5, B, n)
                    else:
                        y_preds_mc = y_preds.unsqueeze(0) # (1, B, n)    
                    dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples, 1), self.params)          # (S * 5 * B, n)
                    dg_dz_all = dg_dz_flat.contiguous().view(self.total_samples, B, n)
                    dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)
                    dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).expand(self.total_samples, -1, -1)   # (S * 5,B,m)
                    inv_matrices_exp   = inv_matrices.unsqueeze(0).expand(self.total_samples, -1, -1, -1)    # (S * 5,B,m,m)
                    w = - torch.einsum('sbi,sbij,sbj->sb',
                            dl_dz_star_pad_exp,
                            inv_matrices_exp,
                            dg_dz_pad) # (S * 5, B)

                    # self.replay_buffer.push(ori_y_preds, x.repeat(self.total_samples, 1), w.detach(), z_star_tensor, info)
                    _y_preds_sample, _x_sample, _w_sample = ori_y_preds, x.repeat(self.total_samples, 1), w.detach()
                else:
                    _y_preds_sample, _x_sample, _w_sample, _, _, _elbo_sample = self.replay_buffer.get_many(idx)

                with torch.enable_grad():
                    time_start = time.time()
                    if epoch % self.interval == 0 or epoch == -1: # sampling from diffusion model
                        _elbo_losses = self.diffusion.compute_elbo_grad_vec(_y_preds_sample, _x_sample)
                    else: # reuse samples from replay buffer
                        _elbo_losses = self.diffusion.compute_elbo_grad_vec(_y_preds_sample.reshape(-1, self.n), _x_sample.repeat_interleave(self.total_samples, dim=0))
                        # import pdb; pdb.set_trace()
                    _elbo_losses = _elbo_losses.contiguous().view(self.total_samples, -1)
                    # B = z_star.size(0)
                    # _elbo_losses = _elbo_losses.view(B, self.total_samples).t().contiguous()
                    time_end = time.time()

                    # print(f"Time for computing elbo = {time_end - time_start}")
                    
                    # Compute weighted ELBO loss and gradients
                    if epoch % self.interval == 0:
                        weighted_elbo = (_w_sample.detach() * _elbo_losses).mean()
                        self.replay_buffer.put(idx, y_pred=y_preds, x=x, weight=w.detach(), z_star=z_star, info=info, elbo=_elbo_losses.detach())
                    elif epoch == -1: # validation, don't push to replay buffer
                        weighted_elbo = (_w_sample.detach() * _elbo_losses).mean()
                    else:
                        is_weight = torch.exp(_elbo_losses - _elbo_sample).detach()
                        is_weight = is_weight.clamp(min=0, max=10)
                        # import pdb; pdb.set_trace()
                        weighted_elbo = (is_weight * _w_sample.detach() * _elbo_losses).mean()
                        print(f"is_weight = {is_weight.mean()}")

                    # weighted_elbo = (w.detach() * _elbo_losses).sum(dim=1).mean() / float(self.total_samples)
                    # weighted_elbo = (_w_sample.detach() * _elbo_losses).mean()

                    # import pdb; pdb.set_trace()
                    
                    # sample_grads = []
                    # for i in range(self.total_samples):
                    #     sample_loss = (w.detach() * _elbo_losses[i]).mean() / self.total_samples
                    #     grads = torch.autograd.grad(
                    #         sample_loss,
                    #         diffusion_params,
                    #         retain_graph=True,
                    #         allow_unused=False,
                    #         create_graph=False
                    #     )
                    #     # Move gradients to CPU before storing
                    #     grads = tuple(g.cpu() for g in grads)
                    #     sample_grads.append(grads)
                    #     del sample_loss, grads
                    
                    # var_per_param = []
                    # for param_idx in range(len(diffusion_params)):
                    #     param_grads = torch.stack([g[param_idx] for g in sample_grads], dim=0)
                    #     var_per_param.append(param_grads.var(dim=0, unbiased=True).mean().item())
                    # print(f"var_per_param = {[f'{x:.1e}' for x in var_per_param]}")
                    
                    weighted_elbo_grads = torch.autograd.grad(
                        weighted_elbo,
                        diffusion_params,
                        retain_graph=False,
                        allow_unused=False,
                        create_graph=False
                    )
                # import pdb; pdb.set_trace()
                time_end = time.time()
                # print(f"Time for backward = {time_end - time_start}")

                return None, None, None, None, *tuple(weighted_elbo_grads)
        
        if epoch % self.interval == 0 or epoch == -1:
            return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor, _x, idx, *_diffusion_params)
        else:
            return DiffusionCvxpyFn.apply(_y_preds_sample, _z_star_sample, _x_sample, idx, *_diffusion_params)