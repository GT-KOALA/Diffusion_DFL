import cvxpy as cp
import numpy as np
import torch
from autograd import jacobian
import autograd.numpy as auto_np
from cvxpylayer_local import utils
from cvxpylayer_local.cvxpylayer import CvxpyLayer
import wandb
import os
from joblib import Parallel, delayed, parallel_backend

# in paper, we call g as f
def dg_dz(y_preds: torch.Tensor, z_star: torch.Tensor, params):
    alpha = params["alpha"]

    if y_preds.dim() == 1:  # (n,)
        s = (y_preds * z_star).sum()
        return alpha * s * y_preds - y_preds

    if y_preds.dim() == 2:  # (m,n)
        s = (y_preds * z_star).sum(dim=-1, keepdim=True)
        return alpha * y_preds * s - y_preds

    if y_preds.dim() == 3:
        if z_star.dim() == 1:            # (n,)
            zs = z_star.view(1, 1, -1)   # (1,1,n) broadcast over (B,M,n)
        elif z_star.dim() == 2 and z_star.shape[0] == y_preds.shape[0]:
            zs = z_star.unsqueeze(1)     # (B,1,n) broadcast over M
        else:
            raise ValueError(f"Bad z_star shape {tuple(z_star.shape)} for y_preds {tuple(y_preds.shape)}")

        s = (y_preds * zs).sum(dim=-1, keepdim=True)         # (B,M,1)
        return alpha * y_preds * s - y_preds                 # (B,M,n)

    raise ValueError(f"Unexpected y_preds shape: {tuple(y_preds.shape)}")

def d2g_dzdy(y_preds, z_star, params):
    """
    y_preds: (B, M, n) or (B, n)  (interpreted as (B,1,n))
    z_star : (B, n)
    returns: (B, n, n) for ∂^2 g / (∂z ∂y)
             where g(z;Y) = (1/M) sum_j [ 0.5*alpha*(y_j^T z)^2 - (y_j^T z) ]
             so  ∂^2 g / (∂z ∂y) = alpha*( mean_y z^T + mean_s * I ) - I
             with mean_y = (1/M)∑ y_j, mean_s = (1/M)∑ (y_j^T z)
    """
    alpha = params["alpha"]

    if y_preds.dim() == 2:                      # (B, n) -> (B, 1, n)
        y_preds = y_preds.unsqueeze(1)
    z_unsq = z_star.unsqueeze(1)                # (B, 1, n)

    # s_j = y_j^T z*
    s = (y_preds * z_unsq).sum(dim=-1)         # (B, M)
    mean_s = s.mean(dim=1)                     # (B,)

    # mean_y = (1/M) sum y_j
    mean_y = y_preds.mean(dim=1)               # (B, n)

    Bsz, n = z_star.shape
    I = torch.eye(n, device=z_star.device, dtype=z_star.dtype).unsqueeze(0).expand(Bsz, n, n)

    # mean_y z^T
    yzT = torch.einsum('bi,bj->bij', mean_y, z_star)  # (B, n, n)

    # alpha (mean_y z^T + mean_s I) − I
    H = alpha * (yzT + mean_s.view(-1, 1, 1) * I) - I
    return H


# def cvxpy_portfolio(params, mc_samples):
#     n = params["n"]
#     alpha = params["alpha"]

#     z_var = cp.Variable(n)
#     if mc_samples == 1:
#         y_param = cp.Parameter(n)                     # (n,)
#         t = y_param @ z_var                          # scalar
#         obj = -t
#     else:
#         y_param = cp.Parameter((mc_samples, n))       # (M, n)
#         t_vec = y_param @ z_var                     # (M,)
#         obj = (0.5 * alpha * cp.sum_squares(t_vec) - cp.sum(t_vec)) / mc_samples

#     constraints = [
#         cp.sum(z_var) == 1,
#         z_var >= 0,
#         z_var <= 1
#     ]
#     problem = cp.Problem(cp.Minimize(obj), constraints)

#     def layer_fn(y_preds):
#         device = y_preds.device
#         B = y_preds.shape[0]

#         if y_preds.ndim == 3:
#             y_np = y_preds.cpu().detach().numpy()
#         else:
#             assert mc_samples == 1
#             y_np = y_preds.cpu().detach().numpy()  # (B, n)

#         z_batch_np = np.zeros((y_np.shape[0], n), dtype=np.float32)
#         inv_list = []

#         total_time = 0
#         for i in range(y_np.shape[0]):
#             y_param.value = y_np[i]
#             # problem.solve(solver=cp.SCS, verbose=False)
#             problem.solve(solver=cp.CLARABEL, verbose=False)
#             total_time += problem.solver_stats.solve_time

#             z_star = z_var.value.copy().astype(np.float32)
#             z_batch_np[i] = z_star
            
#             lam_eq = problem.constraints[0].dual_value  # (1,)
#             lam_eq = np.array([lam_eq],  dtype=np.float64)
#             lam_ineq1 = problem.constraints[1].dual_value  # (n,)
#             lam_ineq1 = np.asarray(lam_ineq1, dtype=np.float64).ravel()
#             lam_ineq2 = problem.constraints[2].dual_value  # (n,)
#             lam_ineq2 = np.asarray(lam_ineq2, dtype=np.float64).ravel()

#             # lam_eq[lam_eq < 1e-10] = 0
#             # lam_ineq1[lam_ineq1 < 1e-10] = 0
#             # lam_ineq2[lam_ineq2 < 1e-10] = 0
#             lam = np.concatenate([lam_ineq1, lam_ineq2], axis=0).astype(np.float64)

#             # TODO
#             if mc_samples == 1:
#                 H  = alpha
#             else:
#                 H  = alpha # (n, n)
#                 Q = np.ones((1, n), dtype=np.float32) # (1, n)
#                 G = np.vstack([np.eye(n), -np.eye(n)]) # (2 * n, n)
#                 h_z = np.concatenate([z_star, 1 - z_star]) # (2 * n,)

#             DG = np.diag(lam) @ G # (2, 1)
#             Dh = np.diag(h_z) # (2, 2)

#             top = np.concatenate([H, G.T, Q.T], axis=1) # (n, n + 2 * n + 1)
#             mid = np.concatenate([DG, Dh, np.zeros((2 * n, 1), dtype=np.float32)], axis=1) # (2 * n, n + 2 * n + 1)
#             bottom = np.concatenate([Q, np.zeros((1,1)), np.zeros((1,2*n), dtype=np.float32)], axis=1) # (1, n + 2 * n + 1)
#             KKT = np.concatenate([top, mid, bottom], axis=0) # (n + 2 * n + 1, n + 2 * n + 1)
#             invKKT = np.linalg.inv(KKT).astype(np.float32) # (n + 2 * n + 1, n + 2 * n + 1)
#             inv_list.append(invKKT)

#             # wandb.log({"time/num_iters": problem.solver_stats.num_iters, "time/solve_time": problem.solver_stats.solve_time})

#         inv_np = np.stack(inv_list, axis=0)           # (B, KKTdim, KKTdim)

#         z_t  = torch.from_numpy(z_batch_np).to(device)
#         info = {"inv_matrices": inv_np}
#         # wandb.log({"time/total_time": total_time})
#         # print(f"Total time: {total_time}")
#         return (z_t,), info

#     return layer_fn


os.environ["PYDEVD_DISABLE_SUBPROCESS_TRACE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def cvxpy_portfolio_parallel_kkt(params, mc_samples: int = 1, n_jobs: int = 32):
    n = params["n"]
    alpha = params["alpha"]
    z_max = 1.0

    def _worker_init():
        _worker_init.Z = cp.Variable(n)

        _worker_init.Y = cp.Parameter((mc_samples, n))

        t_vec = _worker_init.Y @ _worker_init.Z
        obj = (0.5 * alpha * cp.sum_squares(t_vec) - cp.sum(t_vec)) / mc_samples

        cons = [
            cp.sum(_worker_init.Z) == 1,
            -_worker_init.Z <= 0,
            _worker_init.Z <= z_max
        ]
        _worker_init.CON = cons
        _worker_init.PROB = cp.Problem(cp.Minimize(obj), cons)

    def _solve_one(Y_i):
        if not hasattr(_worker_init, "PROB"):
            _worker_init()
        W = _worker_init

        W.Y.value = Y_i
        try:
            W.PROB.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            W.PROB.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-1)

        # status = (W.PROB.status in ("optimal", "optimal_inaccurate")) and (W.Z.value is not None)
        # if not status:
        #     W.PROB.solve(solver=cp.SCS, verbose=True, eps=1e-1)

        try:
            z_star = W.Z.value.astype(np.float64)
        except:
            W.PROB.solve(solver=cp.SCS, verbose=False, eps=1e-1)
            z_star = W.Z.value.astype(np.float64)

        lam_eq    = np.atleast_1d(W.PROB.constraints[0].dual_value).astype(np.float64)  # (1,)
        lam_lower = np.asarray(W.PROB.constraints[1].dual_value, dtype=np.float64).ravel()  # (n,)
        lam_upper = np.asarray(W.PROB.constraints[2].dual_value, dtype=np.float64).ravel()  # (n,)

        H = alpha * (Y_i.T @ Y_i) / mc_samples         # (n, n)
        H = H.astype(np.float64)

        # G = np.vstack([np.eye(n), -np.eye(n)])
        G = np.vstack([-np.eye(n), np.eye(n)])

        h_val = np.concatenate([-z_star, z_star - z_max]).astype(np.float64)    # (2n,)

        lam = np.concatenate([lam_lower, lam_upper]).astype(np.float64)         # (2n,)
        DG  = np.diag(lam) @ G                                                  # (2n, n)
        Dh  = np.diag(h_val)                                                    # (2n, 2n)

        Q = np.ones((1, n), dtype=np.float64)                                    # (1, n)

        top = np.concatenate([H, G.T, Q.T], axis=1) # (n, n + 2 * n + 1)
        mid = np.concatenate([DG, Dh, np.zeros((2 * n, 1), dtype=np.float64)], axis=1) # (2 * n, n + 2 * n + 1)
        bottom = np.concatenate([Q,  np.zeros((1,2*n), dtype=np.float64),np.zeros((1,1), dtype=np.float64)], axis=1) # (1, n + 2 * n + 1)
        KKT = np.concatenate([top, mid, bottom], axis=0) # (n + 2 * n + 1, n + 2 * n + 1)

        eps = 1e-6
        KKT += eps * np.eye(3 * n + 1)
        return z_star.astype(np.float32), KKT.astype(np.float32)

    def layer_fn(y_preds: torch.Tensor):
        device = y_preds.device
        y_np = y_preds.detach().cpu().numpy()

        if y_np.ndim == 2:
            y_np = y_np[:, None, :]
        assert y_np.ndim == 3 and y_np.shape[1] == mc_samples and y_np.shape[2] == n, \
            f"Expected y_preds shape (B,{mc_samples},{n}) or (B,{n}); got {y_preds.shape}"

        with Parallel(n_jobs=n_jobs, backend="loky", initializer=_worker_init) as pool:
            results = pool(delayed(_solve_one)(y_np[i]) for i in range(len(y_np)))

        z_list, KKT_list = map(np.array, zip(*results))  # z_list: (B, n)
        z_t = torch.from_numpy(z_list).to(device)

        info = {"KKT_matrices": np.stack(KKT_list, axis=0)}  # (B, 3n+1, 3n+1)
        return (z_t,), info

    return layer_fn