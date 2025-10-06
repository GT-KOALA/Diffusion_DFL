import torch
import tqdm
import time
from utils import computeCovariance
from data_utils import SP500DataLoader, generateDataset

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import os
import sys


import pandas as pd
import torch
import numpy as np
import qpth
import scipy
import cvxpy as cp
import random
import argparse
import tqdm
import time
import datetime as dt
from cvxpylayers.torch import CvxpyLayer
from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt

import torch.nn
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection
from sqrtm import sqrtm

REG = 0.1
solver = 'cvxpy'

MAX_NORM = 0.1
T_MAX_NORM = 0.1

def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def computeCovariance(covariance_mat):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n = len(covariance_mat)
    cosine_matrix = torch.zeros((n,n))
    for i in range(n):
        cosine_matrix[i] = cos(covariance_mat, covariance_mat[i].repeat(n,1))
    return cosine_matrix

def rmse_loss(mu_pred, Y_actual):
    if mu_pred.ndim == 3:
        return ((mu_pred - Y_actual)**2).mean(dim=(0,1)).sqrt().data.cpu().numpy()
    else:
        return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()

def _dfl_loss(y, z, alpha):
    y_z = (y * z).sum(dim=-1) # (batch_size,)
    return (0.5 * alpha * y_z**2 - y_z).sum()

def _dfl_loss_linear(y, z):
    y_z = (y * z).sum(dim=-1) # (batch_size,)
    return y_z.mean()

def get_grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    grads = [g for g in grads if g is not None]
    return torch.sqrt(sum(g.pow(2).sum() for g in grads) + 1e-12)

def train_portfolio_diffusion(model, layer, optimizer, epoch, dataset, params, device='cpu', evaluate=False):
    model.model_net.train()
    alpha = params["alpha"]
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    x_dim = params["x_dim"]
    y_dim = params["n"]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            # features, covariance_mat, labels = features[0]  .to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, y_dim)
            z_star = layer(features.float()) # (50, 28)
            z_star = z_star.squeeze(1) # (50,)
            dfl_loss = _dfl_loss(labels, z_star, alpha)

            diffusion_loss = 0.0
            num_samples = 10
            for _ in range(num_samples):
                t = torch.randint(0, model.timesteps, (features.shape[0],), device=device).long()
                diffusion_loss += model.diffusion_loss(labels, t, features)
            diffusion_loss /= num_samples

            dfl_grad_norm = get_grad_norm(dfl_loss.mean(), model.model_net.parameters())
            mse_grad_norm = get_grad_norm(diffusion_loss, model.model_net.parameters())
            print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
            gamma = 0.5
            loss = gamma * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - gamma) * diffusion_loss

            if not evaluate:
                optimizer.zero_grad()
                backward_start_time = time.time()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.model_net.parameters(), float('inf'))
                optimizer.step()
                backward_time += time.time() - backward_start_time
            else:
                grad_norm = 0

            train_objs.append(dfl_loss.mean().detach().item())
            # tqdm_loader.set_postfix(obj=f'{obj.detach().item()*100:.2f}%') 
            tqdm_loader.set_postfix(obj=f'{loss.detach().item():.2f}', grad_norm=f'{grad_norm:.2e}') 

    average_loss = 0
    average_obj = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

@torch.no_grad()
def validate_portfolio_diffusion(model, layer, epoch, dataset, params, device='cpu', evaluate=False):
    model.model_net.eval()
    alpha = params["alpha"]
    # covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    x_dim = params["x_dim"]
    y_dim = params["n"]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, y_dim)

            z_star = layer(features.float()) # (50, 1)
            z_star = z_star.squeeze(1) # (50,)

            obj = _dfl_loss_linear(labels, z_star)
 
            validate_objs.append(obj.item())

    average_loss = 0
    average_obj = np.mean(validate_objs)

    return average_loss, average_obj

@torch.no_grad()
def test_portfolio_diffusion(model, layer, epoch, dataset, params, device='cpu', evaluate=False):
    model.model_net.eval()
    alpha = params["alpha"]
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    x_dim = params["x_dim"]
    y_dim = params["n"]
    
    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, y_dim)

            if evaluate:
                layer = cvxpy_portfolio_parallel_kkt(params, mc_samples=1)
                _res = layer(labels)
                z_star = _res[0][0]
                obj = _dfl_loss_linear(labels, z_star.float())
            else:
                z_star = layer(features.float()) # (50, 1)
                z_star = z_star.squeeze(1) # (50,)

                obj = _dfl_loss_linear(labels, z_star)

            test_objs.append(obj.item())

    average_loss = 0
    average_obj = np.mean(test_objs)
    return average_loss, average_obj

def eval_diffusion(which, diffusion, train_loader, test_loader, params, mc_samples, save_folder):
    layer = cvxpy_portfolio_parallel_kkt(params, mc_samples)
    device = diffusion.device
    with torch.no_grad():
        diffusion.model_net.eval()
        train_rmse_list = []
        train_dfl_list = []
        for batch_idx, (features, covariance_mat, labels) in enumerate(train_loader):
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_pred_train = diffusion.sample_elbo(features.repeat(mc_samples, 1), test_mode=True)
            y_pred_train = y_pred_train.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()
            
            train_rmse = rmse_loss(y_pred_train, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            train_rmse_list.append(train_rmse)

            z_star_train, _ = layer(y_pred_train.double())
            z_star_train = z_star_train[0]
            train_loss_task = _dfl_loss_linear(
                labels, z_star_train)
            train_dfl_list.append(train_loss_task.item())
        
        train_rmse = np.mean(train_rmse_list)
        train_dfl = np.mean(train_dfl_list)

        test_rmse_list = []
        test_dfl_list = []
        for batch_idx, (features, covariance_mat, labels) in enumerate(test_loader):
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_pred_test = diffusion.sample_elbo(features.repeat(mc_samples, 1), test_mode=True)
            y_pred_test = y_pred_test.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

            test_rmse = rmse_loss(y_pred_test, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            test_rmse_list.append(test_rmse)
            z_star_test, _ = layer(y_pred_test.double())
            z_star_test = z_star_test[0]
            test_loss_task = _dfl_loss_linear(
                labels, z_star_test)
            test_dfl_list.append(test_loss_task.item())
        test_rmse = np.mean(test_rmse_list)
        test_dfl = np.mean(test_dfl_list)

        with open(
            os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
            np.save(f, train_rmse)
        with open(
            os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
            np.save(f, test_rmse)
        
        torch.save(train_dfl, 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_dfl, 
            os.path.join(save_folder, '{}_test_task'.format(which)))

        print(f"test_dfl = {test_dfl:.4f}")

if __name__ == "__main__":
    from diffusion_opt import Diffsion
    from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt
    from solver_distribution_kkt import DiffusionCvxpyModule as DiffusionCvxpyModule_distr
    from solver_reparam import DiffusionCvxpyModule as DiffusionCvxpyModule_reparam

    portfolio_opt_dir = os.path.abspath(os.path.dirname(__file__))
    print("portfolio_opt_dir:", portfolio_opt_dir)

    n = 50
    mc_samples = 20
    device = "cuda:7"
    
    sp500_data_dir = os.path.join(portfolio_opt_dir, "data", "sp500")
    sp500_data = SP500DataLoader(sp500_data_dir, "sp500",
                                start_date=dt.datetime(2004, 1, 1),
                                end_date=dt.datetime(2017, 1, 1),
                                collapse="daily",
                                overwrite=False,
                                verbose=True)
    
    train_loader, validate_loader, test_loader = generateDataset(sp500_data, n=n, num_samples=1000000, batch_size=64)
    feature_size = train_loader.dataset[0][0].shape[1] * n
    params = {"n": n, "alpha": 1e-6, "x_dim": feature_size}

    # base_save = f'stock_portfolio_results_{n}'
    base_save = os.path.join(YOUR_HOME_FOLDER, "e2e-model-learning", "stock_portfolio", "stock_portfolio_results_50")
    with torch.no_grad():
        for run in range(2, 7):
            save_dir = os.path.join(base_save, f"{run}")
            if not os.path.exists(save_dir):
                raise ValueError(f"Save directory {save_dir} does not exist")

            diffusion_timesteps = 1000
            model = Diffsion(x_dim=feature_size, y_dim=n, device=device, timesteps=diffusion_timesteps)
            model.model_net.load_state_dict(torch.load(os.path.join(save_dir, f"diffusion-point_est-SEED{run}.pth"), weights_only=True))

            layer_reparam = DiffusionCvxpyModule_reparam(model, params, mc_samples, distr_est=False)
            
            eval_diffusion("diffusion-point_est", model, train_loader, test_loader, params, mc_samples, save_dir)
            test_loss, test_obj = test_portfolio_diffusion(model, layer_reparam, 1, test_loader, params, evaluate=False, device=device)
            print(f"test_loss = {test_loss}, test_obj = {test_obj}")

            # model.model_net.load_state_dict(torch.load(os.path.join(save_dir, f"diffusion-distr_est-SEED{run}.pth")))
            # eval_diffusion("diffusion-distr_est", model, train_loader, test_loader, params, mc_samples, save_dir)


    # model = GaussianMLP(feature_size, [1024, 1024], n).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # pretrain_loader, pretrain_validate_loader, pretrain_test_loader = generateDataset(sp500_data, n=n, num_samples=num_samples, batch_size=512)
    # if args.pretrain_epochs > 0:
    #     model = run_rmse_net(model, pretrain_loader, pretrain_validate_loader, pretrain_optimizer, base_save, args, device)
    # layer = MLPCvxpyModule(model, params, args.mc_samples)

    # model = MLP(input_dim=feature_size, hidden_dim=1024, output_dim=n).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # pretrain_loader, pretrain_validate_loader, pretrain_test_loader = generateDataset(sp500_data, n=n, num_samples=num_samples, batch_size=512)
    # if args.pretrain_epochs > 0:
    #     model = run_rmse_net(model, pretrain_loader, pretrain_validate_loader, pretrain_optimizer, base_save, args, device)