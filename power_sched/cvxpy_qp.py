import cvxpy as cp
import numpy as np
import torch
from autograd import jacobian
import autograd.numpy as auto_np
from cvxpylayer_local import utils
from cvxpylayer_local.cvxpylayer import CvxpyLayer

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

def cvxpy_ps(params, mc_samples):
    n = params["n"]
    c_ramp = params["c_ramp"]
    gs, ge = params["gamma_under"], params["gamma_over"]

    z_var = cp.Variable(n)
    if mc_samples == 1:
        y_param = cp.Parameter(n)
    else:
        y_param = cp.Parameter([mc_samples, n])
    
    diff = y_param - z_var

    under = cp.pos(diff)
    over = cp.pos(-diff)
    quad = 0.5 * cp.sum_squares(diff)
    obj = (gs * cp.sum(under) + ge * cp.sum(over) + quad) / float(mc_samples)

    constraints = [
        z_var[1:] - z_var[:-1] <= c_ramp,
        z_var[:-1] - z_var[1:] <= c_ramp
    ]
    problem = cp.Problem(cp.Minimize(obj), constraints)

    def layer_fn(y_preds):
        device = y_preds.device
        B = y_preds.shape[0]

        if y_preds.ndim == 3:
            y_np = y_preds.cpu().detach().numpy()
        else:
            assert mc_samples == 1
            y_np = y_preds.cpu().detach().numpy()  # (B, n)

        z_batch_np = np.zeros((y_np.shape[0], n), dtype=np.float32)
        inv_list = []
        eye_n = np.eye(n, dtype=np.float32)

        for i in range(y_np.shape[0]):
            y_param.value = y_np[i] if mc_samples == 1 else y_np[i]
            problem.solve(solver=cp.SCS)

            z_star = z_var.value.copy().astype(np.float32)
            z_batch_np[i] = z_star

            lam_u = problem.constraints[0].dual_value  # (n-1,)
            lam_o = problem.constraints[1].dual_value  # (n-1,)
            lam = np.concatenate([lam_u, lam_o], axis=0).astype(np.float32)

            dh = jacobian_h_auto(z_star, c_ramp).astype(np.float32)   # (2(n-1), n)
            Dlam = np.diag(lam)
            H  = eye_n
            hdiag = np.diag(constraint_fn(z_star, c_ramp).astype(np.float32))

            top = np.concatenate([H, dh.T], axis=1)          # (n, n+2(n-1))
            bottom = np.concatenate([Dlam.dot(dh), hdiag], axis=1)      # (2(n-1), n+2(n-1))
            _KKT = np.concatenate([top, bottom], axis=0)              # (n+2(n-1), …)
            invKKT = np.linalg.inv(_KKT).astype(np.float32)

            inv_list.append(invKKT)

        inv_np = np.stack(inv_list, axis=0)           # (B, KKTdim, KKTdim)

        z_t  = torch.from_numpy(z_batch_np).to(device)
        info = {"inv_matrices": inv_np}
        return (z_t,), info

    return layer_fn

# This version requires the gradient of solver
def cvxpy_layer(params, mc_samples):
    n = params["n"]
    c_ramp = params["c_ramp"]
    gs = params["gamma_under"]
    ge = params["gamma_over"]
    eps = 0
    Q = cp.Constant(np.eye(n) * (1.0 + eps))
    
    z = cp.Variable(n)
    if mc_samples == 1:
        y_param = cp.Parameter(n)
    else:
        y_param = cp.Parameter([mc_samples, n])
    
    diff = y_param - z
    under = cp.maximum(0, diff)
    over = cp.maximum(0, -diff)
    quad_term = 0.5 * cp.sum_squares(diff)
    obj = (
        gs * cp.sum(under) + ge * cp.sum(over) + quad_term
    ) / float(mc_samples)

    # obj = 0
    # for i in range(mc_samples):
    #     obj += gs * cp.sum(cp.maximum(0, y_param[i, :] - z)) \
    #          + ge * cp.sum(cp.maximum(0, z -y_param[i, :])) \
    #          + 0.5 * cp.sum_squares(z - y_param[i, :])
    # obj = obj / mc_samples

    objective = cp.Minimize(obj)

    constraints = [
        z[1:] - z[:-1] <= c_ramp,
        -z[1:] + z[:-1] <= c_ramp
    ]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp()

    def fwd_with_scs(params_numpy, context):
        inv_matrices = []
        if params_numpy[0].ndim == 2:
            batch_size, n = params_numpy[0].shape
        elif params_numpy[0].ndim == 3:
            batch_size, M, n = params_numpy[0].shape
            
        z_batch  = np.zeros((batch_size, n))
        for b in range(batch_size):
            y_param.value = params_numpy[0][b]
            problem.solve(solver=cp.SCS, requires_grad=False)
            duals_0 = problem.constraints[0].dual_value
            duals_1 = problem.constraints[1].dual_value
            lambdas = np.concatenate([duals_0, duals_1], axis=0)
            z_star = z.value.copy()
            z_batch[b] = z_star

            h_val_z_star = constraint_fn(z_star, c_ramp)
            dh_dz = jacobian_h_auto(z_star, c_ramp)
            diag_lambda = np.diag(lambdas)
            diag_h = np.diag(h_val_z_star)
            matrix = np.concatenate([np.concatenate([Q.value, dh_dz.T], axis=1), 
                                     np.concatenate([diag_lambda @ dh_dz, diag_h], axis=1)], axis=0) # [94, 94]
            inv_matrix = np.linalg.inv(matrix)
            inv_matrices.append(inv_matrix)
        
        sol = [z_batch]
        info = {}
        info["inv_matrices"] = np.stack(inv_matrices, axis=0) # [batch_size, 94, 94]

        return sol, info

    def bwd_with_scs(dvars_numpy, context):
        return problem.backward()

    def fwd(params_numpy, context):
        sol, info = utils.forward_numpy(params_numpy, context)
        z_star = sol[0]

        h_val_z_star = constraint_fn(z_star, c_ramp)
        dh_dz = jacobian_h_auto(z_star, c_ramp)

        lambdas = np.array(info['dual'])

        batch_size = z_star.shape[0]

        inv_matrices = []

        for b in range(batch_size):
            diag_lambda = np.diag(lambdas[b, :46]) # [46, 46]
            diag_h = np.diag(h_val_z_star[b]) # [46, 46]
            matrix = np.concatenate([np.concatenate([Q.value, dh_dz[b].T], axis=1), 
                                     np.concatenate([diag_lambda @ dh_dz[b], diag_h], axis=1)], axis=0) # [94, 94]
            inv_matrix = np.linalg.inv(matrix)
            inv_matrices.append(inv_matrix)
        info["inv_matrices"] = np.stack(inv_matrices, axis=0) # [batch_size, 94, 94]

        return sol, info

    def bwd(dvars_numpy, context):
        return utils.backward_numpy(dvars_numpy, context)

    return CvxpyLayer(problem,
                      parameters=[y_param],
                      variables=[z],
                      custom_method=(fwd_with_scs, bwd))