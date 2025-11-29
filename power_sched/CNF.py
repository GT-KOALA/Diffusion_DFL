import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.datasets
import os
import time

from torchdyn.core import NeuralODE
from torchdiffeq import odeint
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.distributions import MultivariateNormal
from torchdyn.nn import Augmenter
from torchdyn.models.cnf import CNF, hutch_trace, autograd_trace

def mlp(in_dim, hidden_dim, out_dim, n_layers=3):
    layers, d = [], in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.ELU()]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

class _CondVecField(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()
        self.net = mlp(x_dim + y_dim, hidden_dim, x_dim, n_layers=4)
        self._ctx = None  # (B, y_dim)

    def set_context(self, y):
        self._ctx = y

    def forward(self, x, t=None):
        # x: (B, x_dim) ; ctx: (B, y_dim)
        if self._ctx is None:
            raise RuntimeError("Context y is not set. Call set_context(y) before forward.")
        if self._ctx.shape[0] != x.shape[0]:
            # align batch
            y = self._ctx
            if y.shape[0] == 1:
                y = y.repeat(x.shape[0], 1)
            else:
                raise RuntimeError(f"Context batch {y.shape[0]} != x batch {x.shape[0]}")
        else:
            y = self._ctx
        return self.net(torch.cat([x, y], dim=-1))

class CNF_model(nn.Module):
    def __init__(self, x_dim, y_dim, augment_dim=1, hidden_dim=1024, timesteps=100, device="cuda"):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.augment_dim = augment_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps

        self.vec_field = _CondVecField(x_dim, y_dim, hidden_dim)

        noise = MultivariateNormal(loc=torch.zeros(x_dim, device=self.device),
                           covariance_matrix=torch.eye(x_dim, device=self.device))
  
        cnf = CNF(self.vec_field, trace_estimator=hutch_trace, noise_dist=noise)
        self.nde = NeuralODE(cnf, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4, return_t_eval=False)
        self.flow = nn.Sequential(Augmenter(augment_idx=1, augment_dims=self.augment_dim), self.nde).to(self.device)
        self.prior = MultivariateNormal(loc=torch.zeros(x_dim, device=self.device),
                                       covariance_matrix=torch.eye(x_dim, device=self.device))

    def _set_ctx(self, y):
        self.vec_field.set_context(y.to(self.device))


    def train_loss(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self._set_ctx(y)
        out = self.flow(x)        # (B, 1 + x_dim)
        if out.dim() == 3:
            out = out[-1]
        delta_logdet = out[:, 0]
        z = out[:, 1:]
        logprob = self.prior.log_prob(z) - delta_logdet
        return -logprob.mean()

    @torch.no_grad()
    def sample(self, n, y):
        """
        Sample x ~ p_theta(x|y):
          1) z ~ N(0,I)
          2) Inverse flow (base->data) conditioned on y
        y: (y_dim,) or (n, y_dim)
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if y.shape[0] == 1:
            y = y.repeat(n, 1)
        elif y.shape[0] != n:
            raise ValueError(f"y batch {y.shape[0]} must be 1 or n={n}")

        z = self.prior.sample((n,)).to(self.device)
        self._set_ctx(y.to(self.device))

        cnf_mod = self.flow[1].func  # NeuralODE(...).func -> CNF
        old_dir = getattr(cnf_mod, "direction", +1)
        try:
            cnf_mod.direction = -1  # inverse: base -> data
            x_out = self.flow(z)    # (B, 1 + x_dim)
            x_samples = x_out[:, 1:]
        finally:
            cnf_mod.direction = old_dir
        return x_samples

    def pretrain_CNF(self, X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, args, two_stage=False):
        args.pretrain_model_path = os.path.join(base_save, f'cnf_model_pretrained_{args.pretrain_epochs}.pth')
        if os.path.exists(args.pretrain_model_path) and not two_stage:
            self.flow.load_state_dict(torch.load(args.pretrain_model_path, map_location=self.device))
        else:
            print(f"Model not found at {args.pretrain_model_path}, training from scratch")
            print("--------------------------------")
            os.makedirs(base_save, exist_ok=True)
            # pretrain_optimizer = torch.optim.AdamW(self.flow.parameters(), lr=1e-3, weight_decay=1e-4)
            pretrain_optimizer = torch.optim.Adam(self.flow.parameters(), lr=1e-3)
            best_val = float('inf')
            best_state = None
            pretrain_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=512, shuffle=True)
            
            @torch.no_grad()
            def eval_nll():
                self.eval()
                total, n = 0.0, 0
                x_eval, y_eval = X_hold2_.to(self.device), Y_hold2_.to(self.device)
                loss = self.train_loss(x_eval, y_eval)
                total += loss.item() * x_eval.size(0)
                n += x_eval.size(0)
                return total / max(1, n)

            def run_loop(max_epochs=100):
                best, best_ep, no_imp = float('inf'), -1, 0
                for ep in range(1, max_epochs + 1):
                    self.train()
                    t0 = time.time()
                    for xb, yb in pretrain_data_loader:
                        loss = self.train_loss(xb, yb)
                        pretrain_optimizer.zero_grad()
                        loss.backward()
                        pretrain_optimizer.step()
                    if (ep + 1) % 10 == 0:
                        val = eval_nll()
                        print(f"Epoch {ep}, val NLL = {val:.4f}")
                    dt = time.time() - t0
                    print(f"[cCNF] epoch {ep:03d}/{max_epochs}, train loss = {loss.item(): .4f}, {dt:.2f}s")
                    # if val + 1e-6 < best:
                    #     best, best_ep, no_imp = val, ep, 0
                    #     torch.save({"state_dict": self.state_dict(), "val_nll": best, "epoch": ep}, best_path)
                    # else:
                    #     no_imp += 1
                    #     if no_imp >= patience:
                    #         print(f"[cCNF] early stop @ {ep}, best {best_ep}, val={best:.4f}")
                    #         break
                # if os.path.exists(best_path):
                #     ckpt = torch.load(best_path, map_location=self.device)
                #     self.load_state_dict(ckpt["state_dict"])
                #     print(f"[cCNF] loaded best (epoch={ckpt.get('epoch','?')}, val={ckpt.get('val_nll','?'):.4f})")

            run_loop()
            torch.save(self.flow.state_dict(), os.path.join(base_save, f'cnf_model_pretrained_{args.pretrain_epochs}.pth'))
        