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
from util import bger, expandParam, extract_nBatch
import numpy as np
import autograd.numpy as auto_np
from cvxpylayer_local import utils
import autograd 
from functorch import vmap
import torch.nn.functional as F
import time
from cvxpy_powers_sched import cvxpy_ps, cvxpy_layer, dg_dz, cvxpy_ps_parallel
from cvxpy_powers_sched_kkt import cvxpy_ps_slack
import wandb

def flatten_1d(grad_list):
    return torch.cat([g.contiguous().view(-1) for g in grad_list])

def all_layers_cosine(reparam_grad_list, weighted_elbo_grads_list, eps=1e-12):
    g1 = flatten_1d(reparam_grad_list)
    g2 = flatten_1d(weighted_elbo_grads_list)
    return F.cosine_similarity(g1, g2, dim=0, eps=eps)

def layerwise_cosine(reparam_grad_list, weighted_elbo_grads_list, eps=1e-12):
    cos_list = []
    for g1, g2 in zip(reparam_grad_list, weighted_elbo_grads_list):
        cos = F.cosine_similarity(g1.view(-1), g2.view(-1), dim=0, eps=eps)
        cos_list.append(cos.item())

    return cos_list

def log_list(tag: str, values, step=None, commit=True):
    wandb.log({f"{tag}/{i}": v for i, v in enumerate(values)})

class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True):
        super().__init__()
        self.diffusion = diffusion
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        self.total_samples = mc_samples
        # self.cvxpy_layer = cvxpy_ps_parallel(params, self.mc_samples)
        self.cvxpy_layer =  cvxpy_ps_parallel(params, self.mc_samples)
        # self.cvxpy_layer =  cvxpy_ps_slack(params, self.mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        
    def _sample(self, x, test_mode=False):
        x = x.repeat(self.total_samples, 1) # (batch_size * mc_samples * 5, 24)
        time_start = time.time()
        # with torch.no_grad():
        y_preds = self.diffusion.sample_elbo(x, test_mode=False)   # no need for gradient (batch_size * mc_samples, 24)
        time_end = time.time()
        # print(f"Time for sampling = {time_end - time_start}")

        time_start = time.time()
        elbo_losses = self.diffusion.compute_elbo_grad_vec(y_preds, x)
        elbo_losses = elbo_losses.contiguous().view(self.total_samples, -1) # (mc_samples * 5, batch_size)
        time_end = time.time()
        # print(f"Time for computing elbo = {time_end - time_start}")

        # del x
        # torch.cuda.empty_cache()
        # grad_y_preds = self.pretrain_diffusion.compute_elbo_grad(y_preds, x)
        return y_preds, elbo_losses
    
    def forward(self, x, test_mode=False):
        _y_preds, _elbo_losses = self._sample(x, test_mode=False)
        _elbo_losses = _elbo_losses.contiguous().view(self.total_samples, -1) # (mc_samples * 5, batch_size)

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
        print(f"Time for cvxpy optimization = {time_end - time_start}")
        z_star_tensor = z_star_tuple[0]

        diffusion = self.diffusion
        device = diffusion.device

        # _diffusion_params = tuple(p for p in diffusion.model_net.parameters() if p.requires_grad)
        # _diffusion_params = tuple(p for p in diffusion.model_net.parameters() if p.requires_grad and p not in diffusion.model_net.v_head.parameters())
        _diffusion_params = tuple(
            p
            for name, p in diffusion.model_net.named_parameters()
            if p.requires_grad and not name.startswith("v_head")
        )

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, *diffusion_params):
                ctx.save_for_backward(y_preds, z_star, *diffusion_params)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, *diffusion_params = ctx.saved_tensors
                inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)

                B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                dg_dy = - torch.eye(n, device=device)
                dg_dy = dg_dy.unsqueeze(0).expand(B, -1, -1) # (batch_size, n, n)
                dg_dy_padded = F.pad(dg_dy, (0, 0, 0, m - n))

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
                dg_dz_all = dg_dz_flat.reshape(self.total_samples, B, n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))                  # (S * 5, B, m)

                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]

                dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).repeat(self.total_samples, 1, 1)
                inv_matrices_exp   = inv_matrices.unsqueeze(0).repeat(self.total_samples, 1, 1, 1)
                w = - torch.einsum('sbi,sbij,sbj->sb',
                        dl_dz_star_pad_exp,
                        inv_matrices_exp,
                        dg_dz_pad) # (S, B)
                
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
                
                reparam_grad = torch.autograd.grad(
                    y_preds,
                    diffusion_params,
                    grad_outputs=weighted_grad_y_preds,
                    retain_graph=True,
                    allow_unused=False,
                    create_graph=False
                )

                with torch.enable_grad():
                    wandb.log({"_elbo_losses_max": _elbo_losses.max(), "_elbo_losses_min": _elbo_losses.min()})
                    weighted_elbo = (w.detach() * _elbo_losses).mean()

                weighted_elbo_grads = torch.autograd.grad(
                    weighted_elbo,
                    diffusion_params,
                    retain_graph=False,
                    allow_unused=False,
                    create_graph=False
                )
                time_end = time.time()
                print(f"Time for backward = {time_end - time_start}")

                cos_est = all_layers_cosine(reparam_grad, weighted_elbo_grads)
                cos_est_layerwise = layerwise_cosine(reparam_grad, weighted_elbo_grads)
                print(f"Cosine estimate = {cos_est.mean()}")
                print(f"Cosine estimate layerwise = {cos_est_layerwise}, {torch.tensor(cos_est_layerwise).mean()}")
                log_list("cos_est_layerwise", cos_est_layerwise)
                wandb.log({"cos_est_all_layers": cos_est.mean(), "cos_est_layerwise": torch.tensor(cos_est_layerwise).mean()})

                # import pdb; pdb.set_trace()
                del w, weighted_elbo, weighted_grad_y_preds, reparam_grad, y_preds
                torch.cuda.empty_cache()

                return None, None, *tuple(weighted_elbo_grads)

        return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor, *_diffusion_params)

class QP_Layer(torch.nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()

        self.n  = params["n"]
        self.c  = params["c_ramp"]
        n, c    = self.n, self.c
        self.N  = 3 * n
        self.device = device
        self.eps = 1e-6
        # self.solver = QPSolvers.CVXPY
        self.solver = QPSolvers.PDIPM_BATCHED

        Q = torch.zeros(3*n, 3*n, device=device)
        Q[:n, :n] = torch.eye(n, device=device)       
        Q[n:, n:] = self.eps * torch.eye(2*n, device=device)  
        self.register_buffer('Q_base', Q)

        mats = []
        # s >= y - z
        mats.append(torch.cat([
            -torch.eye(n, device=self.device),
            -torch.eye(n, device=self.device),
            torch.zeros(n, n, device=self.device)
        ], dim=1))
        # t >= z - y
        mats.append(torch.cat([
            torch.eye(n, device=self.device),
            torch.zeros(n, n, device=self.device),
            -torch.eye(n, device=self.device)
        ], dim=1))
        # -z <= 0
        mats.append(torch.cat([
            -torch.eye(n, device=self.device),
            torch.zeros(n, n, device=self.device),
            torch.zeros(n, n, device=self.device)
        ], dim=1))
        # -s <= 0
        mats.append(torch.cat([
            torch.zeros(n, n, device=self.device),
            -torch.eye(n, device=self.device),
            torch.zeros(n, n, device=self.device)
        ], dim=1))
        # -t <= 0
        mats.append(torch.cat([
            torch.zeros(n, n, device=self.device),
            torch.zeros(n, n, device=self.device),
            -torch.eye(n, device=self.device)
        ], dim=1))
        # z[i+1]-z[i] <= c and -z[i+1]+z[i] <= c
        M = torch.zeros(n-1, n, device=self.device)
        for i in range(n-1):
            M[i, i+1] = 1
            M[i, i]   = -1
        mats.append(torch.cat([M, torch.zeros(n-1, 2*n, device=self.device)], dim=1))
        mats.append(torch.cat([-M, torch.zeros(n-1, 2*n, device=self.device)], dim=1))

        G_base = torch.vstack(mats)
        self.register_buffer('G_base', G_base)

        self.qp_fn = QPFunction(solver=self.solver,
                                check_Q_spd=False,
                                device=device)

    def forward(self, y: torch.Tensor):
        gs = self.params["gamma_under"]
        ge = self.params["gamma_over"]
        batch_shape = y.shape[:-1]
        n, device = self.n, self.device

        Q = self.Q_base.unsqueeze(0).expand(*batch_shape, 3*n, 3*n)
        G = self.G_base.unsqueeze(0).expand(*batch_shape, self.G_base.size(0), 3*n)

        p_z = -y
        p_s = gs.expand_as(y)
        p_t = ge.expand_as(y)
        p   = torch.cat([p_z, p_s, p_t], dim=-1)

        zeros_n = torch.zeros_like(y)
        ramp = torch.full((*batch_shape, n-1), self.c, device=device)
        h = torch.cat([ -y, y, zeros_n, zeros_n, zeros_n, ramp, ramp], dim=-1)

        A = torch.empty(*batch_shape, 0, 3*n, device=device)
        b = torch.empty(*batch_shape, 0,    device=device)

        z_star = self.qp_fn(Q, p, G, h, A, b)
        return z_star

def diffusion_differeniable_qp_layer(params,
          eps=1e-6,
          solver=QPSolvers.PDIPM_BATCHED,
          maxIter=20,
          verbose=0,
          notImprovedLim=3,
          device="cpu"):
    n  = params["n"]
    c  = params["c_ramp"]
    N  = 3 * n
    eps = 1e-4
    solver = solver

    Q_base = torch.zeros(3*n, 3*n, device=device)
    Q_base[:n, :n] = torch.eye(n, device=device)       
    Q_base[n:, n:] = eps * torch.eye(2*n, device=device)  

    mats = []
    mats.append(torch.cat([
        -torch.eye(n, device=device),
            torch.eye(n, device=device),
            torch.zeros(n, n, device=device)
    ], dim=1))
    # 2) -s <= 0
    mats.append(torch.cat([
        torch.zeros(n, n, device=device),
        -torch.eye(n, device=device),
        torch.zeros(n, n, device=device)
    ], dim=1))
    # 3)  z - t <= y
    mats.append(torch.cat([
        torch.eye(n, device=device),
        torch.zeros(n, n, device=device),
        -torch.eye(n, device=device)
    ], dim=1))
    # 4) -t <= 0
    mats.append(torch.cat([
        torch.zeros(n, n, device=device),
        torch.zeros(n, n, device=device),
        -torch.eye(n, device=device)
    ], dim=1))
    # 5)&6) ramp constraints z[i+1]-z[i] <= c and -z[i+1]+z[i] <= c
    M = torch.zeros(n-1, n, device=device)
    for i in range(n-1):
        M[i, i+1] = 1
        M[i, i]   = -1
    mats.append(torch.cat([M, torch.zeros(n-1, 2*n, device=device)], dim=1))
    mats.append(torch.cat([-M, torch.zeros(n-1, 2*n, device=device)], dim=1))

    G_base = torch.vstack(mats)
    A = torch.empty(0, 3*n, device=device)
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, y, gs, ge):
            # --- build batch versions ---
            batch_shape = y.shape[:-1]
            device     = y.device

            Q = Q_base.to(device)
            G = G_base.to(device)

            p = torch.cat([
                -y,
                gs.expand_as(y),
                ge.expand_as(y)
            ], dim=-1)

            # h = [ -y; 0; y; 0; ramp; ramp ]
            zeros = torch.zeros_like(y)
            ramp  = torch.full((*batch_shape, n-1), c, device=device)
            h     = torch.cat([ -y, y, zeros, zeros, zeros, ramp, ramp], dim=-1)

            # no equality constraints
            A = torch.empty(*batch_shape, 0, N, device=device)
            b = torch.empty(*batch_shape, 0,   device=device)

            nBatch = extract_nBatch(Q, p, G, h, A, b)
            Q, _ = expandParam(Q, nBatch, 3)
            p, _ = expandParam(p, nBatch, 2)
            G, _ = expandParam(G, nBatch, 3)
            h, _ = expandParam(h, nBatch, 2)
            A, _ = expandParam(A, nBatch, 3)
            b, _ = expandParam(b, nBatch, 2)

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
            
            zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                    Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                    eps, verbose, notImprovedLim, maxIter)

            # cache what backward needs
            ctx.save_for_backward(zhats, Q, p, G, h, A, b)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            # print(f"dl_dzhat = {dl_dzhat.shape}")
            # grab everything
            zhats, Q, p, G, h, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b)
            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)

            neq, nineq = ctx.neq, ctx.nineq

            d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)
            d = d.to(device)

            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
            
            dx, _, dlam, dnu = pdipm_b.solve_kkt(
                ctx.Q_LU, d, G, A, ctx.S_LU,
                dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())

            dps = dx
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dlam
            if h_e:
                dhs = dhs.mean(0)
            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
                if A_e:
                    dAs = dAs.mean(0)
                if b_e:
                    dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            if Q_e:
                dQs = dQs.mean(0)
            if p_e:
                dps = dps.mean(0)


            grads = (dQs, dps, dGs, dhs, dAs, dbs)

            return grads

    return QPFunctionFn.apply