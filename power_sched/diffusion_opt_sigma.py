
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
import os
import copy
from resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler
import torch.distributed as dist
from torch_ema import ExponentialMovingAverage

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def gaussian_neg_log_like(x, mean, log_var):
    """Continuous Gaussian NLL per dimension (nats)."""
    return 0.5 * (math.log(2 * math.pi) + log_var + (x - mean) ** 2 / torch.exp(log_var))

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
    elif timesteps == 10 or timesteps == 20:
        scale = 1
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
        out_dim = y_dim * 2
        # self.activation = Swish()
        
        hidden_dim = 1024

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
            nn.Linear(hidden_dim, out_dim)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 16),
            nn.SiLU(),
            nn.Linear(self.time_dim * 16, self.time_dim),
        )

        self.device = device
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # nn.init.zeros_(self.net[-2].weight)
        # nn.init.zeros_(self.net[-2].bias)

        nn.init.kaiming_uniform_(self.net[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t, timesteps, s):
        t_encoded = timestep_embedding(timesteps, dim=self.time_dim)
        t_encoded = self.time_mlp(t_encoded)

        x_t = x_t.float()
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if s.ndim == 1:
            s = s.unsqueeze(0)

        total_input = torch.cat([x_t, t_encoded, s], dim=-1)
        out = self.net(total_input)
        # eps_pred = self.eps_head(out)
        # v = self.v_head(out)
        eps_pred, v = out.chunk(2, dim=-1)
        v = v.clamp(-1.0, 1.0)
        return eps_pred, v

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

        self.lambda_t = (self.beta ** 2) / (
            2.0 * self.post_var * self.alpha * (1.0 - self.bar_alpha)
        )
        self.lambda_t = self.lambda_t.float().to(self.device)
        self.lambda_t[0] = self.lambda_t[1]

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
    def p_sample_iddpm(self, x_t, t, s, clip_denoised=False):
        gaussian_noise = torch.randn_like(x_t).float()
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        shape = (-1, *([1] * (x_t.ndim - 1)))

        # output: reward and next_state
        beta_t  = self.beta[t].view(*shape)
        alpha_t = self.alpha[t].view(*shape)
        bar_a_t = self.bar_alpha[t].view(*shape)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_bar_a_t = torch.sqrt(1. - bar_a_t)

        log_post_var_t = self.log_post_var[t].view(*shape)
        log_beta_t = (beta_t + 1e-20).log()

        eps_pred, v = self.model_net(x_t, t, s)

        frac = (v + 1.0) * 0.5 
        log_sigma2 = frac * log_post_var_t + (1.0 - frac) * log_beta_t
        sigma = torch.exp(0.5 * log_sigma2)
       
        mean_pred = (x_t - beta_t * eps_pred / sqrt_one_minus_bar_a_t) / sqrt_alpha_t

        x_recover = mean_pred + sigma * gaussian_noise * nonzero_mask
        if torch.sum(nonzero_mask) == 0:
            score = - eps_pred / (sqrt_one_minus_bar_a_t + 1e-20)
            return x_recover, score
        else:
            return x_recover
    
    # @torch.no_grad()
    def p_sample_ddim(self, x_t, t, s, eta=0.0):
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        # DDIM sampling (much faster than DDPM)
        eps_pred, v = self.model_net(x_t, t, s)
        
        alpha_t = self.bar_alpha[t].view(-1, *([1] * (x_t.ndim - 1)))
        alpha_t_prev = self.bar_alpha_prev[t].view(-1, *([1] * (x_t.ndim - 1)))
        
        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        # Predict x0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        
        # Direction pointing to xt
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * eps_pred
        
        # Random noise
        noise = torch.randn_like(x_t)
        
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise * nonzero_mask
        if torch.sum(nonzero_mask) == 0:
            one_minus_bar_alpha_t = self.one_minus_bar_alpha[t].view(-1, *([1] * (x_t.ndim - 1)))
            score = - eps_pred / one_minus_bar_alpha_t
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
                        x_t = self.p_sample_ddim(x_t, t, s)
                    else:
                        x_t, score = self.p_sample_ddim(x_t, t, s)
        else:
            with torch.enable_grad():
                for i in reversed(range(0, self.timesteps)):
                    t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                    if i != 0:
                        x_t = self.p_sample_ddim(x_t, t, s)
                    else:
                        x_t, score = self.p_sample_ddim(x_t, t, s)

        # Compute gradient of the sum (this gives us the sum of gradients)
        # diff_grad = torch.autograd.grad((x_final_flat * score.reshape(-1)).sum(), 
        #                         self.model_net.parameters(),
        #                         retain_graph=False)
        
        # return x_t.cpu().detach().numpy(), score.cpu().detach().numpy()
        return x_t, score
    
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
                        x_t = self.p_sample_ddim(x_t, t, s)
                    else:
                        x_t, score = self.p_sample_ddim(x_t, t, s)
            return x_t
        else:
            with torch.enable_grad():
                for i in reversed(range(0, self.timesteps)):
                    t = torch.full((batch_size, ), i, device=self.device, dtype=torch.long)
                    if i != 0:
                        x_t = self.p_sample_ddim(x_t, t, s)
                    else:
                        x_t, score = self.p_sample_ddim(x_t, t, s)

            return x_t
    
    def compute_elbo_grad_vec(self, x_t, s, num_iter=1000, requires_grad=True):
        _num_iter = num_iter
        B, sample_size = x_t.shape
        # t_eval = torch.randint(
        #     0, self.timesteps,
        #     (B, _num_iter), device=self.device, dtype=torch.long
        # )

        t_flat, weights = self.loss_aware_sampler.sample(B * _num_iter, self.device)

        # t_flat = t_eval.flatten()
        s_flat = s.unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, s.size(-1))

        if requires_grad:
            with torch.enable_grad():
                x0 = x_t.detach().clone().unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size).requires_grad_()
                hybrid_loss, vlb_loss = self.train_loss_hybrid(x0, t_flat, s_flat, vb_stop_grad=False)
        elif not requires_grad:
            with torch.no_grad():
                x0 = x_t.detach().clone().unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size).requires_grad_()
                hybrid_loss, vlb_loss = self.train_loss_hybrid(x0, t_flat, s_flat, vb_stop_grad=True)

        self.loss_aware_sampler.update_with_local_losses(t_flat, hybrid_loss.detach())
        hybrid_loss = hybrid_loss * weights.detach()

        elbo_loss = hybrid_loss.view(B, _num_iter)
        elbo_loss = elbo_loss.mean(1)  # (B,)

        return - elbo_loss
    
    def compute_elbo_grad_vec_weighted(self, x_t, s, num_iter=1000, requires_grad=True):
        _num_iter = num_iter
        B, sample_size = x_t.shape
        t_eval = torch.randint(
            1, self.timesteps,
            (B, _num_iter), device=self.device, dtype=torch.long
        )
        noise = torch.randn(B, _num_iter, sample_size, device=self.device)

        t_flat = t_eval.flatten()
        noise_f = noise.reshape(-1, sample_size)
        s_flat = s.unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, s.size(-1))

        if requires_grad:
            with torch.enable_grad():
                # x0 = x_t.detach().clone().unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size).requires_grad_()
                x0 = x_t.unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size)
                x_t_hat = self.q_sample(x0, t_flat, noise=noise_f)
                pred = self.model_net(x_t_hat, t_flat, s_flat)
                loss_vec = F.mse_loss(pred, noise_f, reduction="none")
                lam_f = self.lambda_t[t_flat].view(-1, 1)
                loss_f = (lam_f * loss_vec).sum(dim=1)
        elif not requires_grad:
            with torch.no_grad():
                x0 = x_t.detach().clone().unsqueeze(1).expand(-1, _num_iter, -1).reshape(-1, sample_size)
                x_t_hat = self.q_sample(x0, t_flat, noise=noise_f)
                pred = self.model_net(x_t_hat, t_flat, s_flat)
                loss_vec = F.mse_loss(pred, noise_f, reduction="none")
                lam_f = self.lambda_t[t_flat].view(-1, 1)
                loss_f = (lam_f * loss_vec).sum(dim=1)

        elbo_loss = loss_f.view(B, _num_iter)
        elbo_loss = elbo_loss.mean(1)  # (B,)

        return - elbo_loss

    def compute_elbo_grad(self, x_t, s, num_iter=2000):
        batch_size = 1 if x_t.ndim == 1 else x_t.shape[0]

        with torch.enable_grad():
            x_0 = x_t.clone().detach().float().requires_grad_(True)
            # x_0 = x_t
            _num_iter = num_iter
            avg_elbo_loss = 0
            for _ in range(_num_iter):
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
            avg_elbo_loss /= _num_iter
        return - avg_elbo_loss
    
    def _vb_terms_bpd(
        self, x0, x_t, t, eps_pred, v, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        shape = (-1, *([1] * (x_t.ndim - 1)))      #(B,1) or (B,1,1,1)

        beta_t       = self.beta[t].view(*shape)
        alpha_t      = self.alpha[t].view(*shape)
        bar_alpha_t  = self.bar_alpha[t].view(*shape)
        bar_alpha_prev_t = self.bar_alpha_prev[t].view(*shape)
        log_post_var_t  = self.log_post_var[t].view(*shape)

        # ---------- 1.  compute log_sigma2 ----------
        frac         = (v + 1.) * 0.5
        log_beta_t   = (beta_t + 1e-20).log()
        log_sigma2   = frac * log_post_var_t + (1. - frac) * log_beta_t

        # ---------- 2.  compute mu_theta ----------
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_bar_alpha_t  = (1. - bar_alpha_t).sqrt()
        mu_theta = (x_t - beta_t * eps_pred / sqrt_one_minus_bar_alpha_t) / sqrt_alpha_t

        # ---------- 3.  compute true posterior q(x_{t-1}|x_t,x_0) ----------
        coeff1 = beta_t * bar_alpha_prev_t.sqrt() / (1. - bar_alpha_t)
        coeff2 = sqrt_alpha_t * (1. - bar_alpha_prev_t) / (1. - bar_alpha_t)
        mu_true = coeff1 * x0 + coeff2 * x_t
        # mu_true = (x_t - beta_t * eps_true / sqrt_one_minus_bar_alpha_t) / sqrt_alpha_t
        # eps_from_xt = (x_t - torch.sqrt(bar_alpha_t) * x0) / torch.sqrt(1 - bar_alpha_t)

        kl = normal_kl(
            mu_true, log_post_var_t, mu_theta, log_sigma2
        )
        kl = mean_flat(kl) / math.log(2.0)
        
        decoder_nll = gaussian_neg_log_like(x0, mu_theta, log_sigma2)
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl) # shape (batch_size, )
        return output, mu_theta
    
    def _prior_bpd(self, x0):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        tT = torch.full((x0.size(0),), self.timesteps - 1, device=x0.device, dtype=torch.long)
        mu_q, _, log_var_q = self.q_mean_variance(x0, tT)
        prior = normal_kl(mu_q, log_var_q, 0.0, 0.0)
        return mean_flat(prior) / math.log(2)

    def q_mean_variance(self, x0, t):
        mean = self.bar_alpha[t].sqrt().unsqueeze(1) * x0
        var = (1 - self.bar_alpha[t]).unsqueeze(1)
        return mean, var, var.log()
    
    def calc_bpd_loop(self, x0, s, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x0.device
        batch_size = x0.shape[0]

        vb = []
        mse_x0 = []
        mse_mu = []
        for t in list(range(self.timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t_batch, noise=noise)
            eps_pred, v = self.model_net(x_t, t_batch, s)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                L_vlb_t, mu_theta = self._vb_terms_bpd(x0, x_t, t_batch, eps_pred.detach(), v)
            vb.append(L_vlb_t)
            mse_x0.append(mean_flat((mu_theta - x0) ** 2))
            mse_mu.append(mean_flat((eps_pred - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        mse_x0 = torch.stack(mse_x0, dim=1)
        mse_mu = torch.stack(mse_mu, dim=1)

        prior_bpd = self._prior_bpd(x0)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "mse_x0": mse_x0,
            "mse_mu": mse_mu,
        }
    
    def train_loss_hybrid(self, x0, t, s, vb_stop_grad=True):
        self.model_net.train()
        gaussian_noise = torch.randn_like(x0).float()
        x_t = self.q_sample(x0, t, noise=gaussian_noise)

        # eps_pred: predicted noise, v: predicted sigma
        eps_pred, v = self.model_net(x_t, t, s)

        # simple loss (eps MSE)
        L_simple = F.mse_loss(eps_pred, gaussian_noise, reduction="none").mean(dim=1)

        # if vb_stop_grad:
        #     L_vlb_t, mu_theta = self._vb_terms_bpd(x0, x_t, t, eps_pred.detach(), v)
        # else:
        #     L_vlb_t, mu_theta = self._vb_terms_bpd(x0, x_t, t, eps_pred, v)
        L_vlb_t, mu_theta = self._vb_terms_bpd(x0, x_t, t, eps_pred.detach(), v)

        # _lambda = 1e-3
        # assert L_vlb_t.shape == L_simple.shape
        # hybrid_loss = L_simple + _lambda * L_vlb_t

        return L_simple, L_vlb_t
    
    def pretrain_diffusion(self, X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, args, two_stage=False):
        args.pretrain_model_path = os.path.join(base_save, f'diffusion_model_pretrained_{args.pretrain_epochs}.pth')
        if os.path.exists(args.pretrain_model_path) and not two_stage:
            self.model_net.load_state_dict(torch.load(args.pretrain_model_path, map_location=self.device))
            # ts_eval_loss = 0
            # for batch in hold_loader:
            #     if len(batch) == 2:
            #         x_eval, y_eval = batch
            #     elif len(batch) == 3:
            #         x_eval, y_eval, idx = batch

            #     x_eval, y_eval = x_eval.to(self.device), y_eval.to(self.device)
            #     t = torch.randint(0, self.timesteps, (x_eval.shape[0],), device=self.device).long()
            #     ts_eval_loss += self.train_loss_hybrid(y_eval, t, x_eval)
            # ts_eval_loss /= len(hold_loader)
            # print(f"Pretrain final ts_eval_loss = {ts_eval_loss.item(): .4f}")
        elif args.pretrain_epochs > 0:
            print(f"Pretrain model not found at {args.pretrain_model_path}, training from scratch")
            print("--------------------------------")
            os.makedirs(base_save, exist_ok=True)
            pretrain_optimizer = torch.optim.AdamW(self.model_net.parameters(), lr=1e-3, weight_decay=1e-3)
            ema = ExponentialMovingAverage(self.model_net.parameters(), decay=0.9999)

            best_val = float('inf')
            best_state = None
            pretrain_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=1024, shuffle=True)
            loss_aware_sampler = create_named_schedule_sampler("loss-second-moment", self)

            for epoch in range(args.pretrain_epochs):
                epoch_loss = 0
                for _, (x, y) in enumerate(pretrain_data_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    cur_batch_size = x.shape[0]
                    pretrain_optimizer.zero_grad()
                    t, weights = loss_aware_sampler.sample(cur_batch_size, self.device)
                    hybrid_loss, vlb_loss = self.train_loss_hybrid(y, t, x, vb_stop_grad=True)
                    loss_aware_sampler.update_with_local_losses(t, hybrid_loss.detach())
                    hybrid_loss = (hybrid_loss * weights.detach()).mean()
                    hybrid_loss.backward()
                    pretrain_optimizer.step()
                    ema.update()

                    epoch_loss += hybrid_loss.item()
                epoch_loss /= len(pretrain_data_loader)

                if epoch % 100 == 0:
                    self.model_net.eval()
                    with ema.average_parameters():
                        x_eval, y_eval = X_hold2_.to(self.device), Y_hold2_.to(self.device)
                        bdp = self.calc_bpd_loop(y_eval, x_eval)["total_bpd"].mean().item()
                        print(f"Pretrain Epoch {epoch}, pretrain_loss = {epoch_loss: .4f}, validation loss (bpd) = {bdp: .4f}")

            #             if bdp < best_val:
            #                 best_val = bdp
            #                 best_epoch = epoch
            #                 best_state = copy.deepcopy(self.model_net.state_dict())
            # print(f"Best epoch: {best_epoch}, Best bdp: {best_val: .4f}")

            # self.model_net.load_state_dict(best_state)
            x_eval, y_eval = X_hold2_.to(self.device), Y_hold2_.to(self.device)
            bdp = self.calc_bpd_loop(y_eval, x_eval)["total_bpd"].mean().item()
            print(f"Pretrain final bdp = {bdp: .4f}")
            if not two_stage:
                torch.save(self.model_net.state_dict(), os.path.join(base_save, f'diffusion_model_pretrained_{args.pretrain_epochs}.pth'))
            
if __name__ == "__main__":
    timesteps = 500
    gaussian_diffusion = Diffsion(timesteps=timesteps)
    