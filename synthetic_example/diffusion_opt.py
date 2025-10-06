
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
import os
from resample import create_named_schedule_sampler

def soft_clip(x, L, U, beta=0.8):
    return L + F.softplus(x - L, beta=beta) - F.softplus(x - U, beta=beta)

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the embeddings. Defaults to 10000.

    Returns:
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    if dim == 1:
        return torch.sin(timesteps)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    if timesteps == 1000:
        scale = 1000 / timesteps
    elif timesteps == 2000:
        scale = 2000 / timesteps
    elif timesteps == 5000:
        scale = 5000 / timesteps
    elif timesteps == 50:
        scale = 1
    elif timesteps == 10 or timesteps == 20 or timesteps == 300:
        scale = 1
    else:
        raise ValueError(f"Unknown timesteps: {timesteps}")
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class ModelNet(nn.Module):
    def __init__(self, x_dim, y_dim, device):
        super().__init__()
        self.time_dim = 16
        in_dim = x_dim + y_dim + self.time_dim
        out_dim = y_dim
        # self.activation = Swish()
        
        hidden_dim = 1024
        dropout_prob=0.1

        self.activation = nn.SiLU()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout_prob),
            # nn.Linear(hidden_dim, hidden_dim),
            # self.activation,
            # nn.Linear(hidden_dim, hidden_dim),
            # self.activation,
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

        self.time_linear1 = nn.Linear(self.time_dim, self.time_dim * 16)
        self.time_linear2 = nn.Linear(self.time_dim * 16, self.time_dim)

        self.device = device
    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.net:
    #         if isinstance(m, nn.Linear):
    #             # nn.init.xavier_normal_(m.weight)
    #             # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    #             nn.init.zeros_(m.weight)
    #             nn.init.zeros_(m.bias)

    def forward(self, x_t, timesteps, s):
        t_encoded = timestep_embedding(timesteps, dim=self.time_dim)
        t_encoded = self.time_linear1(t_encoded)
        t_encoded = nn.SiLU()(t_encoded)
        t_encoded = self.time_linear2(t_encoded)

        x_t = x_t.float()
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if s.ndim == 1:
            s = s.unsqueeze(0)

        total_input = torch.cat([x_t, t_encoded, s], dim=-1)
        out = self.net(total_input)
        return out

class Diffsion:
    def __init__(self, x_dim, y_dim, timesteps=1000, beta_schedule="linear", device="cpu"):
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.model_net = ModelNet(x_dim, y_dim, device=self.device).to(device)
        
        self.timesteps = timesteps
        self.beta = None
        if beta_schedule == "linear":
            self.beta = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            self.beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        self.beta = self.beta.float().to(self.device)
        
        self.alpha = (1.0 - self.beta).to(self.device)
        self.bar_alpha = torch.cumprod(self.alpha, dim=0).to(self.device)
        # Coefficient for a_{t-1}; pad with 1 at the start
        self.bar_alpha_prev = F.pad(self.bar_alpha[:-1], (1, 0), value=1.).to(self.device)
        self.one_minus_bar_alpha = torch.sqrt(1 - self.bar_alpha).to(self.device)
        
        # Calculations for the posterior q(x_{t-1} | x_t, x_0)
        self.post_var = self.beta * (1. - self.bar_alpha_prev) / (1. - self.bar_alpha)
        self.post_var = self.post_var.to(self.device)
        self.log_post_var = torch.log(self.post_var.clamp(min=1e-20)).to(self.device)

        self.loss_aware_sampler = create_named_schedule_sampler("loss-second-moment", self)
        
    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        alpha_t = self.bar_alpha[t].view(-1, *([1] * (x_start.ndim - 1)))
        # one_minus_alpha_t = torch.sqrt(1 - self.bar_alpha[t]).view(-1, *([1] * (x_start.ndim - 1)))
        one_minus_bar_alpha_t = self.one_minus_bar_alpha[t].view(-1, *([1] * (x_start.ndim - 1)))
        x_t = torch.sqrt(alpha_t) * x_start + one_minus_bar_alpha_t * noise
        return x_t

    # @torch.no_grad()
    def p_sample(self, x_t, t, s, clip_denoised=False):
        gaussian_noise = torch.randn_like(x_t).float()
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))

        # output: reward and next_state
        beta_t = self.beta[t].view(-1, *([1] * (x_t.ndim - 1)))
        a_t = self.alpha[t].view(-1, *([1] * (x_t.ndim - 1)))
        one_minus_bar_alpha_t = self.one_minus_bar_alpha[t].view(-1, *([1] * (x_t.ndim - 1)))
        log_post_var_t = self.log_post_var[t].view(-1, *([1] * (x_t.ndim - 1)))

        output = self.model_net(x_t, t, s)

        if clip_denoised:
            output = torch.clamp(output, min=-1.0, max=1.0)
       
        mu = (1 / torch.sqrt(a_t)) * (x_t - (beta_t * output) / one_minus_bar_alpha_t)
        sigma = nonzero_mask * torch.exp(0.5 * log_post_var_t)
        x_recover = mu + sigma * gaussian_noise
        if torch.sum(nonzero_mask) == 0:
            score = - output / one_minus_bar_alpha_t
            return x_recover, score
        else:
            return x_recover
    
    # @torch.no_grad()
    def p_sample_ddim(self, x_t, t, s, eta=0.0):
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        # DDIM sampling (much faster than DDPM)
        output = self.model_net(x_t, t, s)
        
        alpha_t = self.bar_alpha[t].view(-1, *([1] * (x_t.ndim - 1)))
        alpha_t_prev = self.bar_alpha_prev[t].view(-1, *([1] * (x_t.ndim - 1)))
        
        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        # Predict x0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * output) / torch.sqrt(alpha_t)
        
        # Direction pointing to xt
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * output
        
        # Random noise
        noise = torch.randn_like(x_t)
        
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise * nonzero_mask
        if torch.sum(nonzero_mask) == 0:
            one_minus_bar_alpha_t = self.one_minus_bar_alpha[t].view(-1, *([1] * (x_t.ndim - 1)))
            score = - output / one_minus_bar_alpha_t
            return x_prev, score
        else:
            return x_prev
    
    # @torch.no_grad()
    def sample(self, s, test_mode=False):
        self.model_net.eval()
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(self.device).float()

        batch_size = 1 if s.ndim == 1 else s.shape[0]
        
        # x_T ~ N(0, I) [reward, next_state]
        if s.ndim == 1:
            # For single state case (1D tensor)
            x_t = torch.randn(batch_size, device=self.device).float()
        else:
            # For batch case (2D tensor)
            x_t = torch.randn(batch_size, self.y_dim, device=self.device).float()

        # sampling_timesteps = 20
        # skip = self.timesteps // sampling_timesteps
        # seq = list(range(1, self.timesteps, skip))

        # with torch.no_grad():
        #     for i in reversed(range(20, self.timesteps)):
        #         t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
        #         x_t = self.p_sample_ddim(x_t, t, s)

        # with torch.enable_grad():
        #     for i in reversed(range(0, 20)):
        #         t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
        #         if i != 0:
        #             x_t = self.p_sample_ddim(x_t, t, s)
        #         else:
        #             x_t, score = self.p_sample_ddim(x_t, t, s)

        if test_mode:
            with torch.no_grad():
                for i in reversed(range(0, self.timesteps)):
                    t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                    if i != 0:
                        x_t = self.p_sample(x_t, t, s)
                    else:
                        x_t, score = self.p_sample(x_t, t, s)
        else:
            with torch.enable_grad():
                for i in reversed(range(0, self.timesteps)):
                    t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                    if i != 0:
                        x_t = self.p_sample(x_t, t, s)
                    else:
                        x_t, score = self.p_sample(x_t, t, s)

        # Compute gradient of the sum (this gives us the sum of gradients)
        # diff_grad = torch.autograd.grad((x_final_flat * score.reshape(-1)).sum(), 
        #                         self.model_net.parameters(),
        #                         retain_graph=False)
        
        # return x_t.cpu().detach().numpy(), score.cpu().detach().numpy()
        return x_t
    
    def sample_elbo(self, s, test_mode=False):
        self.model_net.eval()
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(self.device).float()

        batch_size = 1 if s.ndim == 1 else s.shape[0]
        
        # x_T ~ N(0, I) [reward, next_state]
        if s.ndim == 1:
            # For single state case (1D tensor)
            x_t = torch.randn(batch_size, device=self.device).float()
        else:
            # For batch case (2D tensor)
            x_t = torch.randn(batch_size, self.y_dim, device=self.device).float()

        if test_mode:
            with torch.no_grad():
                for i in reversed(range(0, self.timesteps)):
                    t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                    if i != 0:
                        x_t = self.p_sample_ddim(x_t, t, s, eta=0.8)
                    else:
                        x_t, score = self.p_sample_ddim(x_t, t, s, eta=0.8)
        else:
            with torch.enable_grad():
                for i in reversed(range(0, self.timesteps)):
                    t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                    if i != 0:
                        x_t = self.p_sample_ddim(x_t, t, s, eta=0.8)
                    else:
                        x_t, score = self.p_sample_ddim(x_t, t, s, eta=0.8)
        return x_t
    
    def compute_elbo_grad_vec(self, x_t, s, num_iter=1000, requires_grad=True, resample=False):
        B, sample_size = x_t.shape
        if resample:
            _num_iter = 25
            t_flat, weights = self.loss_aware_sampler.sample(B * _num_iter, self.device)
        else:
            # _num_iter = self.timesteps
            _num_iter = num_iter
            t_eval = (torch.randint(0, self.timesteps, (B, _num_iter), device=self.device, dtype=torch.long))
            t_flat = t_eval.flatten()

        noise = torch.randn(B, _num_iter, sample_size, device=self.device)

        noise_f = noise.reshape(-1, sample_size)
        s_flat = s.unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, s.size(-1))

        if requires_grad:
            with torch.enable_grad():
                x0 = x_t.detach().clone().unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size).requires_grad_()
                x_t_hat = self.q_sample(x0, t_flat, noise=noise_f)
                pred = self.model_net(x_t_hat, t_flat, s_flat)
                # loss_vec = (pred - noise_f).square().sum(-1)
                loss_vec = F.mse_loss(pred, noise_f, reduction="none").sum(dim=1)
        elif not requires_grad:
            with torch.no_grad():
                x0 = x_t.detach().clone().unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size)
                x_t_hat = self.q_sample(x0, t_flat, noise=noise_f)
                pred = self.model_net(x_t_hat, t_flat, s_flat)
                # loss_vec = (pred - noise_f).square().sum(-1)
                loss_vec = F.mse_loss(pred, noise_f, reduction="none").sum(dim=1)

        if resample:
            self.loss_aware_sampler.update_with_local_losses(t_flat, loss_vec.detach())
            loss_vec = loss_vec * weights.detach()
            # import pdb; pdb.set_trace()

        elbo_loss = loss_vec.view(B, _num_iter)
        elbo_loss = elbo_loss.mean(1)  # (B,)

        # if resample:
        #     self.loss_aware_sampler.update_with_local_losses(t_flat, loss_vec.detach())

        #     # Normalize weights
        #     eps = 1e-12
        #     weights = weights / (weights.mean() + eps)
        #     with torch.no_grad():
        #         w_clip = torch.quantile(weights, 0.95)
        #     weights = torch.minimum(weights, w_clip)

        #     # SNIS
        #     w_mat = weights.view(B, _num_iter).detach()
        #     l_mat = loss_vec.view(B, _num_iter)
        #     denom = (w_mat.sum(dim=1, keepdim=True) + eps)
        #     elbo_loss = (w_mat * l_mat).sum(dim=1, keepdim=False) / denom.squeeze(1)  # (B,)

        #     with torch.no_grad():
        #         # ESS over whole mini-batch
        #         w = weights
        #         ess = (w.sum()**2) / (w.square().sum() + eps)
        #         N = w.numel()
        #         target_low  = 0.60 * N
        #         target_high = 0.90 * N
        #         up = 0.02; down = 0.01
        #         if ess < target_low:
        #             self.loss_aware_sampler.uniform_prob = min(0.5, self.loss_aware_sampler.uniform_prob * 1.10 + up)
        #         elif ess > target_high:
        #             self.loss_aware_sampler.uniform_prob = max(0.05, self.loss_aware_sampler.uniform_prob * 0.95 - down)
        # else:
        #     elbo_loss = loss_vec.view(B, _num_iter)
        #     elbo_loss = elbo_loss.mean(1)  # (B,)

        return - elbo_loss

    def compute_elbo_grad_vec_with_grad(self, x_t, s):
        B, x_dim = x_t.shape
        num_iter = 1000
        x0 = x_t.detach().clone().requires_grad_(True)  # [B, x_dim]
        # Expand x0 and s for vectorized computation
        x0_rep = x0.unsqueeze(1).repeat(1, num_iter, 1)  # [B, num_iter, x_dim]
        s_rep = s.unsqueeze(1).repeat(1, num_iter, 1)    # [B, num_iter, s_dim]
        t_eval = torch.randint(0, self.timesteps, (B, num_iter), device=self.device, dtype=torch.long)  # [B, num_iter]
        noise = torch.randn(B, num_iter, x_dim, device=self.device)  # [B, num_iter, x_dim]

        # Flatten for batch processing
        x0_flat = x0_rep.reshape(-1, x_dim)  # [B * num_iter, x_dim]
        s_flat = s_rep.reshape(-1, s.size(-1))                    # [B * num_iter, s_dim]
        t_flat = t_eval.reshape(-1)                               # [B * num_iter]
        noise_flat = noise.reshape(-1, x_dim)                     # [B * num_iter, x_dim]

        with torch.enable_grad():
            x_t_hat = self.q_sample(x0_flat, t_flat, noise=noise_flat)
            pred = self.model_net(x_t_hat, t_flat, s_flat)
            loss_vec = F.mse_loss(pred, noise_flat, reduction="none").sum(dim=1)  # [B * num_iter]
        # Reshape and average over num_iter
        elbo_loss = loss_vec.view(B, num_iter).mean(dim=1)  # [B]

        grad_outputs = torch.ones_like(elbo_loss)
        if elbo_loss.requires_grad:
            grad_x0 = torch.autograd.grad(-elbo_loss, x0, grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]  # [B, x_dim]
        else:
            grad_x0 = None
        return -elbo_loss, grad_x0
    
    def compute_elbo_grad_vec_chunk(self, x_t, s, num_iter=1000, chunk=64):
        B, D = x_t.shape
        elbo_acc = torch.zeros(B, device=x_t.device)

        for start in range(0, num_iter, chunk):
            cur = min(chunk, num_iter - start)

            t_eval  = torch.randint(0, self.timesteps, (B, cur), device=x_t.device)
            noise   = torch.randn(B, cur, D, device=x_t.device)
            x0_rep  = x_t.unsqueeze(1).expand(-1, cur, -1).reshape(-1, D)   # no grad
            s_rep   = s.unsqueeze(1).expand(-1, cur, -1).reshape(-1, s.size(-1))
            t_flat  = t_eval.flatten()
            noise_f = noise.reshape(-1, D)

            with torch.enable_grad():
                x_t_hat = self.q_sample(x0_rep, t_flat, noise=noise_f)
                pred = self.model_net(x_t_hat, t_flat, s_rep)
                loss = F.mse_loss(pred, noise_f, reduction="none").sum(dim=1)

            elbo_acc += loss.view(B, cur).sum(dim=1)

        return -(elbo_acc / num_iter)          # shape (B,)
    
    def compute_elbo_grad(self, x_t, s):
        batch_size = 1 if x_t.ndim == 1 else x_t.shape[0]

        with torch.enable_grad():
            x_0 = x_t.clone().detach().float().requires_grad_(True)
            # x_0 = x_t
            num_iter = 1000
            avg_elbo_loss = 0
            for _ in range(num_iter):
                t_eval = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
                noise = torch.randn_like(x_0).float()
                x_t_hat = self.q_sample(x_0, t_eval, noise=noise)
                pred_noise = self.model_net(x_t_hat, t_eval, s)

                elbo_loss = F.mse_loss(pred_noise, noise, reduction="none").sum(dim=1) # (batch_size, )
                avg_elbo_loss += elbo_loss
                # grad_y_pred = torch.autograd.grad(-elbo_loss.mean(), x_0, retain_graph=True)[0] # grad shape: (batch_size, 24)
                # delbo_dtheta = torch.autograd.grad(x_t, self.model_net.parameters(), grad_outputs=grad_y_pred, retain_graph=True)
                # delbo_dtheta_2 = torch.autograd.grad(-elbo_loss.mean(), self.model_net.parameters(), retain_graph=False)
                # grad_y_pred_list.append(grad_y_pred)

            # avg_grad_y_pred = torch.stack(grad_y_pred_list, dim=0).mean(dim=0)
            # elbo_loss = torch.stack(elbo_loss_list, dim=0).mean(dim=0)
            avg_elbo_loss /= num_iter
        return - avg_elbo_loss

    def sample_fisher(self, s):
        self.model_net.eval()
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(self.device).float()

        batch_size = 1 if s.ndim == 1 else s.shape[0]
        
        # x_T ~ N(0, I) [reward, next_state]
        if s.ndim == 1:
            # For single state case (1D tensor)
            x_t = torch.randn(s.shape[0], device=self.device).float()
        else:
            # For batch case (2D tensor)
            x_t = torch.randn(s.shape[0], s.shape[1], device=self.device).float()

        with torch.no_grad():
            for i in reversed(range(1, self.timesteps)):
                t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, t, s)
        
        x_1 = x_t.requires_grad_(True)
        t_eval = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        
        x_0, score = self.p_sample(x_1, t_eval, s)

        noise = torch.randn_like(x_0).float()
        x_1_prime = self.q_sample(x_0, t_eval, noise=noise)
        pred_noise = self.model_net(x_1_prime, t_eval, s)
        print(self.bar_alpha[0])
        print(self.bar_alpha[999])

        return x_0.detach().cpu().numpy(), (noise - pred_noise) * 10
        
    def diffusion_loss(self, x_start, t, s):
        self.model_net.train()
        gaussian_noise = torch.randn_like(x_start).float()
        # Forward: x_t ~ q(x_t | x_0)
        x_noisy = self.q_sample(x_start, t, noise=gaussian_noise)
        # Reverse: train the noise predicition model condition on s and a
        predicted_noise = self.model_net(x_noisy, t, s)
        loss = F.mse_loss(gaussian_noise, predicted_noise)
        # w = (1 / (1 - self.bar_alpha[t]))
        # sqrt_w = torch.sqrt(w).view(-1, *([1] * (x_start.ndim - 1)))
        # loss = F.mse_loss(sqrt_w * gaussian_noise, sqrt_w * predicted_noise, reduction="mean")
        return loss

    def pretrain_diffusion(self, X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, info):
        data_size, logit_scale, pretrain_epochs, x_dim, y_dim = info["data_size"], info["logit_scale"], info["pretrain_epochs"], info["x_dim"], info["y_dim"]
        pretrain_model_path = os.path.join(base_save, f'diffusion_model_pretrained_{data_size}_{pretrain_epochs}_{logit_scale}_{x_dim}_{y_dim}.pth')
        if os.path.exists(pretrain_model_path):
            self.model_net.load_state_dict(torch.load(pretrain_model_path, map_location=self.device))
            ts_eval_loss = 0
            x_eval, y_eval = X_hold2_.to(self.device), Y_hold2_.to(self.device)
            t = torch.randint(0, self.timesteps, (x_eval.shape[0],), device=self.device).long()
            ts_eval_loss += self.diffusion_loss(y_eval, t, x_eval)
            print(f"Pretrain final ts_eval_loss = {ts_eval_loss.item(): .4f}")
        elif pretrain_epochs == 0:
            print(f"No pretrain")
            print("--------------------------------")
        else:
            print(f"Pretrain model not found at {pretrain_model_path}, training from scratch")
            print("--------------------------------")
            os.makedirs(base_save, exist_ok=True)
            pretrain_optimizer = torch.optim.AdamW(self.model_net.parameters(), lr=1e-3, weight_decay=1e-4)
            best_val = float('inf')
            best_state = None
            pretrain_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=512, shuffle=True)
            for epoch in range(pretrain_epochs):
                self.model_net.train()
                epoch_loss = 0
                for batch in pretrain_data_loader:
                    if len(batch) == 2:
                        x, y = batch
                    elif len(batch) == 3:
                        x, y, idx = batch

                    # for toy example, y1 is random, so we need to shuffle the data
                    perm = torch.randperm(y.size(0), device=y.device)
                    y = y[perm]

                    x, y = x.to(self.device), y.to(self.device)
                    cur_batch_size = x.shape[0]
                    pretrain_optimizer.zero_grad()
                    t = torch.randint(0, self.timesteps, (cur_batch_size,), device=self.device).long()
                    ts_loss = self.diffusion_loss(y, t, x)
                    epoch_loss += ts_loss.item()
                    ts_loss.backward()
                    pretrain_optimizer.step()

                epoch_loss /= len(pretrain_data_loader)
                if epoch % 100 == 0:
                    self.model_net.eval()
                    with torch.no_grad():
                        x_eval, y_eval = X_hold2_.to(self.device), Y_hold2_.to(self.device)
                        t = torch.randint(0, self.timesteps, (x_eval.shape[0],), device=self.device).long()
                        ts_eval_loss = self.diffusion_loss(y_eval, t, x_eval)
                    print(f"Pretrain Epoch {epoch}, pretrain_loss = {epoch_loss: .4f}, validation loss = {ts_eval_loss: .4f}")

            x_eval, y_eval = X_hold2_.to(self.device), Y_hold2_.to(self.device)
            t = torch.randint(0, self.timesteps, (x_eval.shape[0],), device=self.device).long()
            ts_eval_loss = self.diffusion_loss(y_eval, t, x_eval)
            print(f"Pretrain final ts_eval_loss = {ts_eval_loss.item(): .4f}")
            torch.save(self.model_net.state_dict(), os.path.join(base_save, f'diffusion_model_pretrained_{data_size}_{pretrain_epochs}_{logit_scale}_{x_dim}_{y_dim}.pth'))
if __name__ == "__main__":
    timesteps = 500
    gaussian_diffusion = Diffsion(timesteps=timesteps)
    