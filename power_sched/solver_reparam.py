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
import autograd 
from autograd import jacobian
import torch.nn.functional as F
import copy
import time
from cvxpy_powers_sched import cvxpy_ps, cvxpy_ps_parallel
from model_classes import MLP, GaussianMLP

# batch constraint_fn
def constraint_fn(z, c_ramp):
    h1 = z[..., 1:] - z[..., :-1] - c_ramp
    h2 = -z[..., 1:] + z[..., :-1] - c_ramp

    return auto_np.concatenate([h1, h2], axis=-1)

# batch jacobian of constraint_fn
def jacobian_h_auto(z, c_ramp):
    if z.ndim == 1:
        return jacobian(lambda zi: constraint_fn(zi, c_ramp))(z)
    else:
        B, d = z.shape
        single_jac = jacobian(lambda zi: constraint_fn(zi, c_ramp))
        return np.stack([single_jac(z[i]) for i in range(B)], axis=0)

def mul_grad(x, grads):
    for grad in grads:
        x = x * grad.T
    return x

def dg_dz_elbo_weighted(Y_mc_list, Z, grad_elbo_list, gs, ge, Q):
    # Y_mc: (batch_size, mc_samples, 24)
    # Z: (batch_size, 24)
    # elbo_losses: (batch_size, mc_samples)
    # gs, ge: scalar
    # Q: (24, 24)
    if isinstance(Q, np.ndarray):
        Q = torch.from_numpy(Q).float().to(Z.device)

    grad_z = None
    for i in range(len(Y_mc_list)):
        under = ((Y_mc_list[i] - Z) > 0).float() * grad_elbo_list[i]
        over  = ((Z - Y_mc_list[i]) > 0).float() * grad_elbo_list[i]
        quad_term  =  Z @ Q # nothing to do with y
        mean_term  = -(grad_elbo_list[i] * Y_mc_list[i])

        if grad_z is None:
            grad_z = gs * under + ge * over + quad_term + mean_term
        else:
            grad_z += gs * under + ge * over + quad_term + mean_term
    grad_z /= len(Y_mc_list)
    return grad_z

# input single sample of y_pred and z_star
def dg_dz(y_pred, z_star, gs, ge):
    under = - gs * ((y_pred - z_star) > 0).float()
    over = ge * ((z_star - y_pred) > 0).float()
    quad_term = z_star - y_pred
    return under + over + quad_term

def comp_w_i(dl_dz_star_padded, inv_matrices, dg_dz_padded):
    _res = - torch.einsum('bi,bij,bj->b',
                dl_dz_star_padded,
                inv_matrices,
                dg_dz_padded) # (batch_size, )
    return _res


class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion
        # self.pretrain_diffusion = copy.deepcopy(self.diffusion)
        self.mc_samples = mc_samples
        # self.cvxpy_layer = cvxpy_ps(params, mc_samples)
        self.cvxpy_layer = cvxpy_ps_parallel(params, mc_samples)
        self.gs = params["gamma_under"]
        self.ge = params["gamma_over"]
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.mc_samples, 1) # (batch_size * mc_samples, 24)

        y_preds = self.diffusion.sample_elbo(x, test_mode=test_mode)   # (batch_size * mc_samples, 24)
        
        return y_preds
    
    def forward(self, x, test_mode=False):
        time_start = time.time()
        try:
            y_preds = self._sample(x, test_mode=test_mode)
        except RuntimeError as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            raise e
        time_end = time.time()
        # print(f"Time taken for sampling: {time_end - time_start} seconds")
        if self.mc_samples > 1:
            y_preds = y_preds.contiguous().view(self.mc_samples, -1, self.n) # (mc_samples, batch_size, 24)
            y_preds = y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples, 24)

        y_preds_copy = y_preds.clone().detach()

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(y_preds_copy)
        time_end = time.time()
        # print(f"Time taken for cvxpy layer: {time_end - time_start} seconds")
        z_star_tensor = z_star_tuple[0]

        diffusion = self.diffusion
        device = diffusion.device
        distr_est = self.distr_est
    
        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star):
                ctx.save_for_backward(y_preds, z_star)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                y_preds, z_star = ctx.saved_tensors
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

        return DiffusionCvxpyFn.apply(y_preds, z_star_tensor)

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