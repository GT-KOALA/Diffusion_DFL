import torch
import tqdm
import time
from utils import computeCovariance

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

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

def train_portfolio(model, covariance_model, optimizer, epoch, dataset, params, training_method='two-stage', device='cpu', evaluate=False):
    model.train()
    alpha = params["alpha"]
    covariance_model.train()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)
            Q = covariance_model() * (1 - REG) + torch.eye(n) * REG  # TODO

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = sqrtm(Q) # torch.cholesky(Q)
                # =============== solving QP using qpth ================
                if solver == 'qpth':
                    G = -torch.eye(n)
                    h = torch.zeros(n)
                    A = torch.ones(1,n)
                    b = torch.ones(1)
                    qp_solver = qpth.qp.QPFunction()
                    x = qp_solver(alpha * Q, -p, G, h, A, b)[0]
                # =============== solving QP using CVXPY ===============
                elif solver == 'cvxpy':
                    z_var = cp.Variable(n)
                    # L_para = cp.Parameter((n,n))
                    y_preds_var = cp.Parameter(n)
                    constraints = [z_var >= 0, z_var <= 1, cp.sum(z_var) == 1]
                    objective = cp.Minimize(alpha * cp.sum_squares(y_preds_var @ z_var) - cp.sum(y_preds_var @ z_var))
                    problem = cp.Problem(objective, constraints)

                    cvxpylayer = CvxpyLayer(problem, parameters=[y_preds_var], variables=[z_var])
                    z_star, = cvxpylayer(p) # (50)

                obj =  - labels @ z_star + alpha * (z_star @ labels) ** 2

                inference_time += time.time() - inference_start_time
                # ======= opt ===
                # p_opt = labels
                # L_opt = torch.cholesky(Q_real)
                # x_opt, = cvxpylayer(L_opt, p_opt)
                # opt = labels @ x_opt - alpha * x.t() @ Q_real @ x
                # print('obj:', obj, 'opt:', opt)
            else:
                obj = torch.Tensor([0])

            # ====================== back-prop =====================
            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'two-stage':
                    Q_loss = torch.norm(Q - Q_real)
                    (loss + Q_loss).backward()
                elif training_method == 'decision-focused':
                    (-obj).backward()
                    # (-obj + loss).backward() # TODO
                    for parameter in model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    for parameter in covariance_model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("no grad is backpropagated...")
                pass
            optimizer.step()
            backward_time += time.time() - backward_start_time

            train_losses.append(loss.item())
            train_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.2f}', obj=f'{obj.item()*100:.2f}%') 

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def validate_portfolio(model, covariance_model, scheduler, epoch, dataset, params, training_method='two-stage', device='cpu', evaluate=False):
    alpha = params["alpha"]
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]

            loss = loss_fn(predictions, labels)

            if evaluate:
                Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 

                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = sqrtm(Q) # torch.cholesky(Q)
                # =============== solving QP using qpth ================
                if solver == 'qpth':
                    G = -torch.eye(n)
                    h = torch.zeros(n)
                    A = torch.ones(1,n)
                    b = torch.ones(1)
                    qp_solver = qpth.qp.QPFunction()
                    x = qp_solver(alpha * Q, -p, G, h, A, b)[0]
                # =============== solving QP using CVXPY ===============
                elif solver == 'cvxpy':
                    z_var = cp.Variable(n)
                    # L_para = cp.Parameter((n,n))
                    y_preds_var = cp.Parameter(n)
                    constraints = [z_var >= 0, z_var <= 1, cp.sum(z_var) == 1]
                    objective = cp.Minimize(alpha * cp.sum_squares(y_preds_var @ z_var) - cp.sum(y_preds_var @ z_var))
                    problem = cp.Problem(objective, constraints)

                    cvxpylayer = CvxpyLayer(problem, parameters=[y_preds_var], variables=[z_var])
                    z_star, = cvxpylayer(p) # (50)

                obj =  alpha * (z_star @ labels) ** 2 - labels @ z_star

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            validate_losses.append(loss.item())
            validate_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.2f}', obj=f'{obj.item()*100:.2f}%')

    average_loss    = np.mean(validate_losses)
    average_obj     = np.mean(validate_objs)

    if (epoch > 0):
        if training_method == "two-stage":
            scheduler.step(average_loss)
        elif training_method == "decision-focused" or training_method == "surrogate":
            scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def test_portfolio(model, covariance_model, epoch, dataset, params, device='cpu', evaluate=False):
    alpha = params["alpha"]
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG

            if epoch == -1:
                predictions = labels
                Q = Q_real
            else:
                predictions = model(features.float())[:,0]
                Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 

            loss = loss_fn(predictions, labels)

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = sqrtm(Q) # torch.cholesky(Q)
                # =============== solving QP using qpth ================
                if solver == 'qpth':
                    G = -torch.eye(n)
                    h = torch.zeros(n)
                    A = torch.ones(1,n)
                    b = torch.ones(1)
                    qp_solver = qpth.qp.QPFunction()
                    x = qp_solver(alpha * Q, -p, G, h, A, b)[0]
                    # x_opt = qp_solver(alpha * Q_real, -labels, G, h, A, b)[0]
                # =============== solving QP using CVXPY ===============
                elif solver == 'cvxpy':
                    z_var = cp.Variable(n)
                    # L_para = cp.Parameter((n,n))
                    y_preds_var = cp.Parameter(n)
                    constraints = [z_var >= 0, z_var <= 1, cp.sum(z_var) == 1]
                    objective = cp.Minimize(alpha * cp.sum_squares(y_preds_var @ z_var) - cp.sum(y_preds_var @ z_var))
                    problem = cp.Problem(objective, constraints)

                    cvxpylayer = CvxpyLayer(problem, parameters=[y_preds_var], variables=[z_var])
                    z_star, = cvxpylayer(p) # (50)

                obj =  alpha * (z_star @ labels) ** 2 - labels @ z_star
                # opt = labels @ x_opt - alpha * x_opt.t() @ Q_real @ x_opt
                # print('obj:', obj, 'opt:', opt)

                inference_time += time.time() - inference_start_time
                # ======= opt ===
                # p_opt = labels
                # L_opt = torch.cholesky(Q_real)
                # x_opt, = cvxpylayer(L_opt, p_opt)
                # opt = labels @ x_opt - alpha * x_opt.t() @ Q_real @ x_opt
                # test_opts.append(opt.item())
            else:
                obj = torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.2f}', obj=f'{obj.item()*100:.2f}%') 

    # print('opts:', test_opts)
    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)