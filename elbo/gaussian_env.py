import numpy as np
import torch

class GaussianEnv:
    def __init__(self, state_dim=2, action_dim=2, noise_std=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noise_std = noise_std
        # self.W = W[:state_dim, :state_dim+action_dim]
        self.W = np.random.randn(state_dim, state_dim + action_dim).astype(np.float32)

    def mean(self, s, a):
        """Compute the Gaussian mean mu(s,a) = W^T [s; a]."""
        # s: (batch, state_dim), a: (batch, action_dim)
        if isinstance(s, np.ndarray):
            sa = np.concatenate([s, a], axis=-1)  # (batch, state_dim+action_dim)
            return sa.dot(self.W.T)  # (batch, state_dim)
        elif isinstance(s, torch.Tensor):
            sa = torch.cat([s, a], axis=-1)  # (batch, state_dim+action_dim)
            device = s.device
            W_tensor = torch.from_numpy(self.W).to(device=device)
            return sa @ W_tensor.T  # (batch, state_dim)

    def sample(self, s, a):
        mu = self.mean(s, a)
        noise = np.random.randn(*mu.shape) * self.noise_std
        return mu + noise
    
    def log_prob(self, s, a, x):
        mu = self.mean(s, a)                              
        var = self.noise_std ** 2
        diff = x - mu                                   
        d = self.state_dim
        log_det = d * np.log(2 * np.pi * var)            
        quad_form = torch.sum(diff**2, axis=-1) / var       
        return -0.5 * (log_det + quad_form)               # (batch,)

    def grad_log_prob(self, s, a, x):
        mu = self.mean(s, a)
        var = self.noise_std**2
        # nabla log N(x; mu, var I) = -(x - mu) / var
        return -(x - mu) / var


class GaussianEnv_1_cond:
    def __init__(self, x_dim=16, y_dim=16, noise_std=0.1):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.noise_std = noise_std
        # self.W = W[:state_dim, :state_dim]
        self.W = np.random.randn(y_dim, x_dim).astype(np.float32)

    def mean(self, s):
        """Compute the Gaussian mean mu(s) = W^T s."""
        # s: (batch, x_dim)
        if isinstance(s, np.ndarray):
            return s.dot(self.W.T)  # (batch, y_dim)
        elif isinstance(s, torch.Tensor):
            device = s.device
            W_tensor = torch.from_numpy(self.W).to(device=device)
            return s @ W_tensor.T  # (batch, y_dim)

    def sample(self, s):
        mu = self.mean(s)
        noise = np.random.randn(*mu.shape) * self.noise_std
        return mu + noise
    
    def log_prob(self, s, x):
        mu = self.mean(s)                              
        var = self.noise_std ** 2
        diff = x - mu                                   
        d = self.y_dim
        log_det = d * np.log(2 * np.pi * var)            
        quad_form = torch.sum(diff**2, axis=-1) / var       
        return -0.5 * (log_det + quad_form)               # (batch,)

    def grad_log_prob(self, s, x):
        mu = self.mean(s)
        var = self.noise_std**2
        # nabla log N(x; mu, var I) = -(x - mu) / var
        return -(x - mu) / var
