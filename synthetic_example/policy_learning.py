# offline full-information policy learning
import torch
import torch.nn as nn
import torch.optim as optim
import operator
from functools import reduce

import batch
from constants import *

from cvxpy_toy_kkt import task_loss
import model_classes

def _dfl_loss(z_star, y_true, params):
    loss = task_loss(z_star, y_true, params).sum()
    return loss

def rmse_loss(mu_pred, Y_actual):
    return ((mu_pred - Y_actual)**2).mean().sqrt().item()

def run_policy_net(X_train, Y_train, X_test, Y_test, params):
    x_dim = X_train.shape[1]
    model = model_classes.PolicyNet(x_dim, params)
    step_size = 1e-5

    device = X_train.device
    model = model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float, device=device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float, device=device)
    Y_test_t  = torch.tensor(Y_test, dtype=torch.float, device=device)

    opt = optim.Adam(model.parameters(), lr=step_size)

    train_size = X_train.shape[0]
    num_epochs = 300
    best_test_cost = float('inf')
    best_epoch = 0
    for i in range(num_epochs):
        model.train()
        train_cost, rmse = batch_train(train_size, i, X_train_t, Y_train_t, model, opt, _dfl_loss, params)
        model.eval()
        test_cost = batch.get_cost(X_test_t.shape[0], i, model, X_test_t, Y_test_t, _dfl_loss, params)
        if test_cost < best_test_cost:
            best_test_cost = test_cost
            best_epoch = i
            
        if i % 100 == 0:
            print("Epoch", i, "rmse", rmse, "train_cost", train_cost, "test_cost", test_cost)
            
    print("Best test_cost", best_test_cost, "at epoch", best_epoch)
    print("Final train_cost", train_cost)
    print("Final rmse", rmse)
    return model


def batch_train(batch_sz, epoch, X_train_t, Y_train_t, model, opt, cost_fn, params):
    train_cost = 0
    batch_data_, batch_targets_ = \
        batch.get_vars(batch_sz, X_train_t, Y_train_t)
    size = batch_sz

    for i in range(0, X_train_t.size(0), batch_sz):
        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_, batch_targets_ = \
                batch.get_vars(size, X_train_t, Y_train_t)
        
        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]

        opt.zero_grad()
        preds = model(batch_data_)
        batch_cost = cost_fn(preds, batch_targets_, params)
        batch_cost.backward()
        opt.step()

        train_cost += (batch_cost.item() - train_cost) * size / (i + size)

    rmse = rmse_loss(model(X_train_t), Y_train_t)
    return train_cost, rmse