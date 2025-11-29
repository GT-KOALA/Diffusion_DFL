import torch
import tqdm
import time
from utils import computeCovariance
import os

import numpy as np
import cvxpy as cp
from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt
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
from model import GaussianMLP, MLP, PortfolioPolicyNet
import torch.nn as nn

import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection
from sqrtm import sqrtm


# alpha = 2
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

def generateDataset(data_loader, n=200, num_samples=100):
    feature_mat, target_mat, feature_cols, covariance_mat, target_name, dates, symbols = data_loader.load_pytorch_data()
    symbol_indices = np.random.choice(len(symbols), n, replace=False)
    feature_mat    = feature_mat[:num_samples,symbol_indices] # (2898, 50, 28)
    target_mat     = target_mat[:num_samples,symbol_indices] # (2898, 50, 1)
    covariance_mat = covariance_mat[:num_samples,symbol_indices] # (2898, 50, 9)
    symbols = [symbols[i] for i in symbol_indices]
    dates = dates[:num_samples] # (2898,)

    num_samples = len(dates)

    sample_shape, feature_size = feature_mat.shape, feature_mat.shape[-1]

    # ------ normalization ------
    feature_mat = feature_mat.reshape(-1,feature_size)
    feature_mat = (feature_mat - torch.mean(feature_mat, dim=0)) / (torch.std(feature_mat, dim=0) + 1e-5) 
    feature_mat = feature_mat.reshape(sample_shape, feature_size)

    dataset = data_utils.TensorDataset(feature_mat, covariance_mat, target_mat)

    indices = list(range(num_samples))
    # np.random.shuffle(indices)

    train_size, validate_size = int(num_samples * 0.7), int(num_samples * 0.1)
    train_indices    = indices[:train_size]
    validate_indices = indices[train_size:train_size+validate_size]
    test_indices     = indices[train_size+validate_size:]

    batch_size = 1
    train_loader    = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validate_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validate_indices))
    test_loader     = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    # train_dataset    = dataset[train_indices]
    # validate_dataset = dataset[validate_indices]
    # test_dataset     = dataset[test_indices]

    return train_loader, validate_loader, test_loader

def run_rmse_net(model, pretrain_loader, pretrain_validate_loader, pretrain_optimizer, base_save, args, device):
    n = args.n
    x_dim = 28 * n
    for i in range(100):
        for batch_idx, (features, covariance_mat, labels) in enumerate(pretrain_loader):
            # features, covariance_mat, labels = features[0].float().to(device), covariance_mat[0].float().to(device), labels[0,:,0].float().to(device) # only one single data
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, n)
            # n = len(covariance_mat)
    
            pretrain_optimizer.zero_grad()
            if isinstance(model, GaussianMLP):
                model.train()
                mu, logvar = model(features)
                sigma = torch.exp(logvar)
                loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
                train_loss = loss_fn(mu, labels, sigma)
            elif isinstance(model, MLP) or isinstance(model, PortfolioPolicyNet):
                model.train()
                train_loss = nn.MSELoss()(
                    model(features), labels)
            else:
                raise ValueError('Not implemented')
            train_loss.backward()
            pretrain_optimizer.step()

        if isinstance(model, GaussianMLP):
            model.eval()
            mu, logvar = model(features)
            sigma = torch.exp(logvar)
            loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
            test_loss = loss_fn(mu, labels, sigma)
        elif isinstance(model, MLP) or isinstance(model, PortfolioPolicyNet):
            model.eval()
            test_loss = nn.MSELoss()(
                model(features), labels)
        else:
            raise ValueError('Not implemented')
        if i % 200 == 0:
            print(i, train_loss.item(), test_loss.item())

    torch.save(model.state_dict(), f'{base_save}/pretrained_model_{args.task}_{args.pretrain_epochs}.pth')
    return model

def rmse_loss(mu_pred, Y_actual):
    if mu_pred.ndim == 3:
        return ((mu_pred - Y_actual)**2).mean(dim=(0,1)).sqrt().data.cpu().numpy()
    else:
        return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()

def _dfl_loss(y, z, alpha):
    y_z = (y * z).sum(dim=-1)
    return (0.5 * alpha * y_z**2 - y_z).sum()

def _dfl_loss_linear(y, z):
    y_z = (y * z).sum(dim=-1) # (batch_size,)
    return y_z.mean()

def get_grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    grads = [g for g in grads if g is not None]
    return torch.sqrt(sum(g.pow(2).sum() for g in grads) + 1e-12)

def train_portfolio_mlp(model, layer, optimizer, epoch, dataset, params, device='cpu', evaluate=False):
    model.train()
    alpha = params["alpha"]
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            # features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            features, covariance_mat, labels = features.to(device), covariance_mat.to(device), labels.to(device).float() # only one single data
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])
            n = len(covariance_mat)
            y_preds, z_star = layer(features.float()) # (50, 28)
            z_star = z_star.squeeze(1) # (50,)
            dfl_loss = _dfl_loss(labels, z_star.float(), alpha)
            # loss = dfl_loss

            mse_loss = torch.nn.MSELoss()(y_preds.mean(dim=1), labels)
            mse_grad_norm = get_grad_norm(mse_loss, model.parameters())
            dfl_grad_norm = get_grad_norm(dfl_loss, model.parameters())
            print(f"mse_grad_norm = {mse_grad_norm.item():.4f}, dfl_grad_norm = {dfl_grad_norm.item():.4f}")
            gamma = 0.9
            loss = gamma * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss + (1 - gamma) * mse_loss

            optimizer.zero_grad()
            backward_start_time = time.time()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            backward_time += time.time() - backward_start_time

            train_objs.append(dfl_loss.mean().detach().item())
            # tqdm_loader.set_postfix(obj=f'{obj.detach().item()*100:.2f}%') 
            tqdm_loader.set_postfix(obj=f'{dfl_loss.mean().detach().item():.2f}', grad_norm=f'{grad_norm:.2e}') 

    average_loss    = 0
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

@torch.no_grad()
def validate_portfolio_mlp(model, layer, epoch, dataset, params, device='cpu', evaluate=False):
    model.eval()
    alpha = params["alpha"]
    # covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            # features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_preds, z_star = layer(features.float()) # (50, 1)
            z_star = z_star.squeeze(1) # (50,)

            obj = _dfl_loss_linear(labels, z_star.float())
 
            validate_objs.append(obj.item())

    average_loss    = 0
    average_obj     = np.mean(validate_objs)

    return average_loss, average_obj

@torch.no_grad()
def test_portfolio_mlp(model, layer, epoch, dataset, params, device='cpu', evaluate=False):
    model.eval()
    alpha = params["alpha"]
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            # features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])
            if evaluate:
                layer = cvxpy_portfolio_parallel_kkt(params, mc_samples=1)
                _res = layer(labels)
                z_star = _res[0][0]
                obj = _dfl_loss_linear(labels, z_star.float())
            else:
                y_preds, z_star = layer(features.float()) # (50, 1)
                z_star = z_star.squeeze(1) # (50,)
                obj = _dfl_loss_linear(labels, z_star.float())

            test_objs.append(obj.item())

    average_loss    = 0
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj

def eval_net(which, model, train_loader, test_loader, params, save_folder, mc_samples=None):
    if isinstance(model, GaussianMLP):
        solver = cvxpy_portfolio_parallel_kkt(params, mc_samples)
        device = next(model.parameters()).device

        with torch.no_grad():
            model.eval()
            train_rmse_list = []
            train_dfl_list = []
            for batch_idx, (features, covariance_mat, labels) in enumerate(train_loader):
                features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
                features = features.reshape(-1, params["x_dim"])
                labels = labels.reshape(-1, params["n"])

                mu_pred_train, logvar_pred_train = model(features.repeat(mc_samples, 1))
                sig_pred_train = torch.exp(0.5 * logvar_pred_train)

                y_preds_train = mu_pred_train + sig_pred_train * torch.randn_like(mu_pred_train)
                y_preds_train = y_preds_train.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

                train_rmse = rmse_loss(y_preds_train, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
                train_rmse_list.append(train_rmse)

                z_star_train, _ = solver(y_preds_train.double())
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

            mu_pred_test, logvar_pred_test = model(features.repeat(mc_samples, 1))
            sig_pred_test = torch.exp(0.5 * logvar_pred_test)

            y_preds_test = mu_pred_test + sig_pred_test * torch.randn_like(mu_pred_test)
            y_preds_test = y_preds_test.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

            test_rmse = rmse_loss(y_preds_test, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            test_rmse_list.append(test_rmse)

            z_star_test, _ = solver(y_preds_test.double())
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
        print(f"train_dfl = {train_dfl:.4f}, train_rmse = {train_rmse:.4f}, test_dfl = {test_dfl:.4f}, test_rmse = {test_rmse:.4f}")
    elif isinstance(model, MLP):
        mc_samples = 1
        solver = cvxpy_portfolio_parallel_kkt(params, 1)
        device = next(model.parameters()).device

        with torch.no_grad():
            model.eval()
            train_rmse_list = []
            train_dfl_list = []
            for batch_idx, (features, covariance_mat, labels) in enumerate(train_loader):
                features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
                features = features.reshape(-1, params["x_dim"])
                labels = labels.reshape(-1, params["n"])

                y_preds_train = model(features.repeat(mc_samples, 1))
                y_preds_train = y_preds_train.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

                train_rmse = rmse_loss(y_preds_train, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
                train_rmse_list.append(train_rmse)

                z_star_train, _ = solver(y_preds_train.double())
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

                y_preds_test = model(features.repeat(mc_samples, 1))
                y_preds_test = y_preds_test.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

                test_rmse = rmse_loss(y_preds_test, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
                test_rmse_list.append(test_rmse)

                z_star_test, _ = solver(y_preds_test.double())
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
            
            print(f"train_dfl = {train_dfl:.4f}, train_rmse = {train_rmse:.4f}, test_dfl = {test_dfl:.4f}, test_rmse = {test_rmse:.4f}")
    elif isinstance(model, PortfolioPolicyNet):
        device = next(model.parameters()).device

        with torch.no_grad():
            model.eval()
            train_rmse_list = []
            train_dfl_list = []
            for batch_idx, (features, covariance_mat, labels) in enumerate(train_loader):
                features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
                features = features.reshape(-1, params["x_dim"])
                labels = labels.reshape(-1, params["n"])

                y_preds_train = model(features)

                train_rmse = rmse_loss(y_preds_train, labels)
                train_rmse_list.append(train_rmse)

                train_loss_task = _dfl_loss_linear(
                    labels, y_preds_train)
                train_dfl_list.append(train_loss_task.item())
            
            train_rmse = np.mean(train_rmse_list)
            train_dfl = np.mean(train_dfl_list)

            test_rmse_list = []
            test_dfl_list = []
            for batch_idx, (features, covariance_mat, labels) in enumerate(test_loader):
                features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
                features = features.reshape(-1, params["x_dim"])
                labels = labels.reshape(-1, params["n"])

                y_preds_test = model(features)

                test_rmse = rmse_loss(y_preds_test, labels)
                test_rmse_list.append(test_rmse)

                test_loss_task = _dfl_loss_linear(
                    labels, y_preds_test)
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
            print(f"train_dfl = {train_dfl:.4f}, train_rmse = {train_rmse:.4f}, test_dfl = {test_dfl:.4f}, test_rmse = {test_rmse:.4f}")
    else:
        raise ValueError(f"Model type {type(model)} not supported")