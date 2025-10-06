import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from autograd import jacobian
import autograd.numpy as auto_np
from cvxpylayer_local import utils
from cvxpylayer_local.cvxpylayer import CvxpyLayer
import wandb
import time
from functools import partial
from joblib import Parallel, delayed, parallel_backend
import os

def task_loss(Y_sched, Y_actual, params):
    _upper_F = upper_F(Y_sched, Y_actual)
    return _upper_F.mean()

def task_loss_no_mean(Y_sched, Y_actual, params):
    _upper_F = upper_F(Y_sched, Y_actual)
    return _upper_F

def d2g_dzdy(y_preds, z_star, params):
    """
    y_preds: (B, M, n) or (B, n)  (interpreted as (B,1,n))
    z_star : (B, n)
    returns: (B, n, n) for d^2 g / (dz dy)
             where g(z;Y) = (1/M) sum_j exp(- y_j^T z)
    """
    if y_preds.dim() == 2:                      # (B, n) -> (B, 1, n)
        y_preds = y_preds.unsqueeze(1)
    z_star_unsq = z_star.unsqueeze(1)           # (B, 1, n)

    # a_j = exp(- y_j^T z*)
    a = torch.exp(-(y_preds * z_star_unsq).sum(dim=-1))      # (B, M)

    mean_a = a.mean(dim=1)                                     # (B,)
    v = (y_preds * a.unsqueeze(-1)).mean(dim=1)                # (B, n)

    Bsz, n = z_star.shape
    I = torch.eye(n, device=z_star.device, dtype=z_star.dtype).unsqueeze(0).expand(Bsz, n, n)
    # rank1 = torch.einsum('bi,bj->bij', v, z_star)              # (B, n, n)
    rank1 = torch.einsum('bi,bj->bij', z_star, v)

    d2g_dzy = -mean_a.view(-1, 1, 1) * I + rank1
    return d2g_dzy

def dg_dz(y_pred, z_star, params):
    """
    y_pred: (M, n)
    z_star: (n,)
    """
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)  # (1, n)
    # weights_j = exp(- y_j^T z)
    weights = torch.exp(- (y_pred * z_star).sum(dim=-1, keepdim=True))   # (M, 1)
    grad = -(y_pred * weights)
    return grad

os.environ["PYDEVD_DISABLE_SUBPROCESS_TRACE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def cvxpy_toy_parallel_kkt(params, mc_samples=1, n_jobs=1):
    n = params["n"]
    C = params["C"]

    def _worker_init():
        _worker_init.Z = cp.Variable(n)
        _worker_init.Y = cp.Parameter((mc_samples, n))

        scores = _worker_init.Y @ _worker_init.Z                   # (M,)
        obj = cp.sum(cp.exp(-scores)) / float(mc_samples)          # (convex)

        con_lower = (_worker_init.Z >= 0)                          # -z <= 0
        con_upper = (_worker_init.Z <= C)                          #  z <= C
        _worker_init.CON  = (con_lower, con_upper)
        _worker_init.PROB = cp.Problem(cp.Minimize(obj), list(_worker_init.CON))

    def _solve_one(y_pred_np):
        if not hasattr(_worker_init, "PROB"):
            _worker_init()
        W = _worker_init

        if y_pred_np.ndim == 1:
            y_pred_np = y_pred_np.reshape(1, -1)  # (1, n)
        assert y_pred_np.shape == (mc_samples, n), \
            f"Expected (mc_samples={mc_samples}, n={n}), got {y_pred_np.shape}"

        W.Y.value = y_pred_np
        try:
            W.PROB.solve(solver=cp.MOSEK, warm_start=True, verbose=False)
        except cp.SolverError:
            W.PROB.solve(solver=cp.SCS, eps=1e-3, verbose=False)
        if W.Z.value is None:
            W.PROB.solve(solver=cp.SCS, eps=1e-1, verbose=False)

        z_star = np.asarray(W.Z.value, dtype=np.float32)     # (n,)
        lower_bound = 1e-7
        if np.any(z_star < lower_bound):
            z_star[z_star < lower_bound] = 0.0

        # Hessian at z*: H = (1/M) sum_j [ exp(- y_j^T z*) * y_j y_j^T ]
        Y = W.Y.value.astype(np.float64)                               # (M, n)
        s = Y @ z_star.astype(np.float64)                              # (M,)
        a = np.exp(-s) / float(mc_samples)                             # (M,)
        H = Y.T @ (Y * a[:, None])                                     # (n, n)

        G = np.vstack([-np.eye(n, dtype=np.float64),
                        np.eye(n, dtype=np.float64)])              # (2n, n)
        h = np.concatenate([np.zeros(n, dtype=np.float64),
                            C * np.ones(n, dtype=np.float64)])     # (2n,)

        # Duals for the two inequality blocks
        lam_lower = np.asarray(W.CON[0].dual_value, dtype=np.float64).reshape(-1)  # (n,)
        lam_upper = np.asarray(W.CON[1].dual_value, dtype=np.float64).reshape(-1)  # (n,)
        lam = np.concatenate([lam_lower, lam_upper])                                # (2n,)
        Dlam = np.diag(lam)

        # Slack in the convention S = G z - h
        h_val = (G @ z_star.astype(np.float64) - h)                 # (2n,) = [-z; z - B]
        Dslack = np.diag(h_val)

        top    = np.concatenate([H,                G.T], axis=1)        # (n, n+2n)
        bottom = np.concatenate([Dlam @ G,         Dslack], axis=1)     # (2n, n+2n)
        KKT    = np.concatenate([top, bottom], axis=0)                  # (n+2n, n+2n) = (3n,3n)

        return z_star, KKT

    def layer_fn(y_preds):
        """
        y_preds: (B, mc_samples, n)  or  (B, n)
        returns: (z_tensor,), {"KKT_matrices": np.ndarray of shape (B, 2n+1, 2n+1)}
        """
        if isinstance(y_preds, torch.Tensor):
            device = y_preds.device
            y_np = y_preds.detach().cpu().numpy().astype(np.float64)
        else:
            y_np = y_preds
            device = "cpu"

        if y_np.ndim == 2:             # (B, n) -> use one MC sample per item
            y_np = y_np[:, None, :]    # (B, 1, n)

        with Parallel(n_jobs=n_jobs, backend="loky", initializer=_worker_init) as pool:
            results = pool(delayed(_solve_one)(y_np[i]) for i in range(len(y_np)))

        z_np, KKT_list = map(np.array, zip(*results))   # z_np: (B, n)
        return (torch.from_numpy(z_np).to(device),), {"KKT_matrices": np.stack(KKT_list, 0)}

    return layer_fn


def upper_F(z_star, y_true):
    # z: (B,1)
    z = z_star.view(-1, 1) if z_star.dim() == 1 else z_star
    # if z.dim() == 2 and z.shape[1] != 1:
    #     raise ValueError("z_star last dim should be n=1 for this setup")

    if y_true.dim() == 1:
        if y_true.numel() == z.shape[0]:           # (B,1,1)
            y = y_true.view(-1,1).unsqueeze(1)
        else:                                       # (1,1,n)
            y = y_true.view(1,1,-1)
    elif y_true.dim() == 2:
        y = y_true.unsqueeze(1) if y_true.shape[0] == z.shape[0] else y_true.unsqueeze(0)
    elif y_true.dim() == 3:
        y = y_true
    else:
        raise ValueError("Unsupported y_true dim")

    B, M, n = y.shape
    z_b = z.unsqueeze(1).expand(B, M, n)           # (B,M,n)
    val = torch.exp(-(y * z_b).sum(-1)).mean(dim=1)  # (B,)
    return val

