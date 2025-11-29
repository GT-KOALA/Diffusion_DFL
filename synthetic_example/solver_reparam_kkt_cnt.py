import torch
import torch.nn.functional as F
import time
from cvxpy_toy_kkt import cvxpy_toy_parallel_kkt, d2g_dzdy
from model_classes import MLP, GaussianMLP

class CNFCvxpyModule(torch.nn.Module):
    def __init__(self, cnf, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.cnf = cnf
        self.mc_samples = mc_samples
        # self.cvxpy_layer = cvxpy_toy_parallel_kkt(params, mc_samples * params["batch_size"])
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.mc_samples, 1) # (batch_size * mc_samples, 24)

        y_preds = self.cnf.sample(x, test_mode=test_mode)   # (batch_size * mc_samples, 24)
        # print(f"y_preds.max = {y_preds.max()}, y_preds.min = {y_preds.min()}")
        return y_preds
    
    def forward(self, x, _y=None, test_mode=False):
        if _y is not None:
            _y_preds = _y
        else:
            time_start = time.time()
            _y_preds = self._sample(x, test_mode=test_mode)
            time_end = time.time()
            # print(f"Time taken for sampling: {time_end - time_start}")
            
        if self.mc_samples > 1:
            _y_preds = _y_preds.view(self.mc_samples, -1, self.n) # (mc_samples, batch_size, 24)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous() # (batch_size, mc_samples, 24)

        _y_preds_copy = _y_preds.clone().detach()

        time_start = time.time()
        with torch.no_grad():
            cvxpy_layer = cvxpy_toy_parallel_kkt(self.params, x.shape[0] * self.mc_samples)
            z_star_tuple, info = cvxpy_layer(_y_preds_copy.reshape(1, -1, self.n))
        time_end = time.time()
        # print(f"Time taken for cvxpy layer: {time_end - time_start}")
        z_star_tensor = z_star_tuple[0]

        cnf = self.cnf
        device = cnf.device
    
        class CNFCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star):
                ctx.save_for_backward(y_preds, z_star)
                return z_star
            
            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star = ctx.saved_tensors
                # inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device) # (batch_size, 94, 94)
                kkt = torch.from_numpy(info["KKT_matrices"]).float().to(x.device)

                B, m = kkt.shape[0], kkt.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n)) # (batch_size, 3)
                # z_star_unsq = z_star.unsqueeze(1) # (batch_size, 1, 24)
                
                _d2g_dzy = d2g_dzdy(y_preds, z_star, self.params)
                d2g_dzy_padded = F.pad(_d2g_dzy, (0, 0, 0, m - n))

                k_inv_dg = torch.linalg.solve(kkt, d2g_dzy_padded)
                
                weighted_grad_y_preds_list = []
                for _ in range(self.mc_samples):
                    # Point estimation
                    dl_dy = - torch.einsum('bi,bij->bj', dl_dz_star_padded, k_inv_dg)
                    weighted_grad_y_preds_list.append(dl_dy)

                if self.mc_samples > 1:
                    weighted_grad_y_preds = torch.stack(weighted_grad_y_preds_list, dim=1) # (batch_size, mc_samples, 24)
                    weighted_grad_y_preds = weighted_grad_y_preds / float(self.mc_samples)
                else:
                    weighted_grad_y_preds = weighted_grad_y_preds_list[0]

                time_end = time.time()
                # print(f"Time taken for backward: {time_end - time_start}")

                # import pdb; pdb.set_trace()

                return weighted_grad_y_preds, None

        return CNFCvxpyFn.apply(_y_preds, z_star_tensor)