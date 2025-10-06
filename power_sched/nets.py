#/usr/bin/env python3

import os
import numpy as np
import wandb
import time
import gc, traceback
import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import copy

import model_classes
import constants
from diffusion_opt import Diffsion
from cvxpy_powers_sched import cvxpy_ps_parallel
from solver_distribution import DiffusionCvxpyModule as DiffusionCvxpyModule_distr
from solver_reparam import DiffusionCvxpyModule as DiffusionCvxpyModule_reparam
from solver_distribution_replay import DiffusionCvxpyModuleReplay
# from ffoqp_diffusion_module import ffoqp_diffusion
# from qpthlocal.qp import QPFunction, QPSolvers
from util import ReplayBuffer, ReplayBufferW, IndexedDataset
import pickle
from collections import OrderedDict

def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
            0.5 * (Y_sched - Y_actual)**2).mean(0)

def task_loss_no_mean(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
        params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
        0.5 * (Y_sched - Y_actual)**2)

def rmse_loss(mu_pred, Y_actual):
    if mu_pred.ndim == 3:
        return ((mu_pred - Y_actual)**2).mean(dim=(0,1)).sqrt().data.cpu().numpy()
    else:
        return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()

def rmse_loss_weighted(mu_pred, Y_actual, weights):
    return ((weights * (mu_pred - Y_actual)**2).mean(dim=0).sqrt()).sum()


def run_rmse_net(model, variables, X_train, Y_train, training_steps=1000, lr=1e-3, set_sig=False, which=None, save_folder=None):
    if isinstance(model, Diffsion):
        opt = optim.Adam(model.model_net.parameters(), lr=lr)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)

    total_training_time = 0.0
    train_dfl_losses = []
    val_dfl_losses = []
    test_dfl_losses = []
    total_training_time_list = []

    for i in range(training_steps):
        epoch_start_time = time.time()
        opt.zero_grad()
        if isinstance(model, model_classes.GaussianMLP):
            model.train()
            mu, logvar = model(variables['X_train_'])
            sigma = torch.exp(logvar)
            loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
            train_loss = loss_fn(mu, variables['Y_train_'], sigma)
            if save_folder is not None:
                train_dfl_loss = _dfl_loss(variables['Y_train_'], model.sample_elbo(variables['X_train_'], test_mode=True), model, params, constants.DEVICE)
                train_dfl_losses.append(train_dfl_loss.item())
        else:
            model.train()
            if set_sig:
                train_loss = nn.MSELoss()(
                    model(variables['X_train_'])[0], variables['Y_train_'])
            else:
                train_loss = nn.MSELoss()(
                    model(variables['X_train_']), variables['Y_train_'])
            if save_folder is not None:
                train_dfl_loss = _dfl_loss(variables['Y_train_'], model(variables['X_train_']), model, params, constants.DEVICE)
                train_dfl_losses.append(train_dfl_loss.item())
        train_loss.backward()
        opt.step()
        total_training_time += time.time() - epoch_start_time

        if isinstance(model, model_classes.GaussianMLP):
            model.eval()
            mu, logvar = model(variables['X_test_'])
            sigma = torch.exp(logvar)
            loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
            test_loss = loss_fn(mu, variables['Y_test_'], sigma)
            if save_folder is not None:
                test_dfl_loss = _dfl_loss(variables['Y_test_'], model.sample_elbo(variables['X_test_'], test_mode=True), model, params, constants.DEVICE)
                val_dfl_loss = _dfl_loss(variables['Y_test_'], model.sample_elbo(variables['X_test_'], test_mode=True), model, params, constants.DEVICE)
        else:
            model.eval()
            if set_sig:
                test_loss = nn.MSELoss()(
                    model(variables['X_test_'])[0], variables['Y_test_'])
            else:
                test_loss = nn.MSELoss()(
                    model(variables['X_test_']), variables['Y_test_'])
            if save_folder is not None:
                test_dfl_loss = _dfl_loss(variables['Y_test_'], model(variables['X_test_']), model, params, constants.DEVICE)
                val_dfl_loss = _dfl_loss(variables['Y_test_'], model(variables['X_test_']), model, params, constants.DEVICE)
                
        if save_folder is not None:
            train_dfl_losses.append(train_dfl_loss.item())
            val_dfl_losses.append(val_dfl_loss.item())
            test_dfl_losses.append(test_dfl_loss.item())
            total_training_time_list.append(time.time() - epoch_start_time)

        if i % 200 == 0:
            print(i, train_loss.item(), test_loss.item())

    if set_sig and not isinstance(model, Diffsion):
        model.eval()
        model.set_sig(variables['X_train_'], variables['Y_train_'])

    if save_folder is not None:
        run_res = {
            'train_dfl_losses': train_dfl_losses,
            'val_dfl_losses': val_dfl_losses,
            'test_dfl_losses': test_dfl_losses,
            'total_training_time_list': total_training_time_list
        }
        torch.save(run_res, os.path.join(save_folder, f'{which}'))

    return model

def run_weighted_rmse_net(X_train, Y_train, X_test, Y_test, params):
    weights = torch.ones(Y_train.shape, device=constants.DEVICE)
    for i in range(10):
        model, weights2 = run_weighted_rmse_net_helper(X_train, Y_train, X_test, Y_test, params, weights, i)
        weights = weights2.detach()
    return model

def run_weighted_rmse_net_helper(X_train, Y_train, X_test, Y_test, params, weights, i):
    X_train_ = torch.tensor(X_train[:,:-1], dtype=torch.float, device=constants.DEVICE)
    Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=constants.DEVICE)
    X_test_ = torch.tensor(X_test[:,:-1], dtype=torch.float, device=constants.DEVICE)
    Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=constants.DEVICE)

    model = model_classes.Net(X_train[:,:-1], Y_train, [200, 200])
    if constants.USE_GPU:
        model = model.to(constants.DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    solver = model_classes.SolveScheduling(params)
    for j in range(100):
        model.train()
        batch_train_weightrmse(100, i*100 + j, X_train_.data, Y_train_.data, model, opt, weights.data)

    # Rebalance weights
    model.eval()
    mu_pred_train, sig_pred_train = model(X_train_)
    Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
    weights2 = task_loss_no_mean(
        Y_sched_train.float(), Y_train_, params)
    if constants.USE_GPU:
        weights2 = weights2.to(constants.DEVICE)
    model.set_sig(X_train_, Y_train_)

    return model, weights2

def batch_train_weightrmse(batch_sz, epoch, X_train_t, Y_train_t, model, opt, weights_t):

    batch_data_ = torch.empty(batch_sz, X_train_t.size(1), device=constants.DEVICE)
    batch_targets_ = torch.empty(batch_sz, Y_train_t.size(1), device=constants.DEVICE)
    batch_weights_ = torch.empty(batch_sz, weights_t.size(1), device=constants.DEVICE)

    size = batch_sz

    for i in range(0, X_train_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_ = torch.empty(size, X_train_t.size(1), device=constants.DEVICE)
            batch_targets_ = torch.empty(size, Y_train_t.size(1), device=constants.DEVICE)
            batch_weights_ = torch.empty(size, weights_t.size(1), device=constants.DEVICE)

        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]
        batch_weights_.data[:] = weights_t[i:i+size]

        opt.zero_grad()
        preds = model(batch_data_)[0]

        ((batch_weights_ * (preds - batch_targets_)**2).mean(dim=0).sqrt()).sum().backward()

        opt.step()

        print ('Epoch: {}, {}/{}'.format(epoch, i+batch_sz, X_train_t.size(0)))

def run_task_net(model, variables, params, X_train, Y_train, args):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    solver = model_classes.SolveScheduling(params)

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(1000):
        opt.zero_grad()
        model.train()
        mu_pred_train, sig_pred_train = model(variables['X_train_'])
        Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
        train_loss = task_loss(
            Y_sched_train.float(),variables['Y_train_'], params)
        train_loss.sum().backward()

        model.eval()
        mu_pred_test, sig_pred_test = model(variables['X_test_'])
        Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
        test_loss = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)

        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])
        Y_sched_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
        hold_loss = task_loss(
            Y_sched_hold.float(), variables['Y_hold_'], params)

        opt.step()

        print(i, train_loss.sum().item(), test_loss.sum().item(), 
            hold_loss.sum().item())
        
        wandb.log({"train_loss": train_loss.sum().item(), "test_loss": test_loss.sum().item(), "hold_loss": hold_loss.sum().item()})

        # Early stopping
        hold_costs.append(hold_loss.sum().item())
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()
                best_model = model_classes.Net(
                    X_train[:,:-1], Y_train, [200, 200])
                best_model.load_state_dict(model_states[idx])
                if constants.USE_GPU:
                    best_model = best_model.to(constants.DEVICE)
                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model

def run_task_net_dataloader(
        model,
        train_loader,
        hold_loader,
        test_loader,
        params,
        max_epochs = 20,
        num_stop_rounds = 10,
        which=None,
        save_folder=None,
):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver = model_classes.SolveScheduling(params)

    best_state, best_hold = None, float('inf')
    stop_window = []

    train_losses = []
    test_losses = []
    hold_losses = []
    total_training_time_list = []
    total_training_time = 0.0

    def evaluate(loader, mode="test"):
        """
        Accepts either a DataLoader **or** a (X, Y) tuple of tensors.
        Returns mean loss (already summed inside task_loss, so divide by N).
        """
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                mu, sig = model(xb)
                y_sched  = solver(mu.double(), sig.double())
                loss_val = task_loss(y_sched.float(), yb, params).sum()
                total += loss_val.item()
                n += 1
            final = total / n
            if mode == "hold":
                print(f"Epoch {epoch}: hold_loss={final:.4f}")
                wandb.log({"hold_loss": final})
            elif mode == "test":
                print(f"Epoch {epoch}: test_loss={final:.4f}")
                wandb.log({"test_loss": final})
        return final

    # --- 3. training loop -----------------------------------------------------------------------
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        for xb, yb in train_loader:
            opt.zero_grad()
            mu, sig = model(xb)
            y_sched = solver(mu.double(), sig.double())
            loss = task_loss(y_sched.float(), yb, params).sum()
            train_loss += loss.item()
            loss.backward()
            opt.step()
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}")
            wandb.log({"train_loss": loss.item()})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        total_training_time += time.time() - epoch_start_time
        total_training_time_list.append(total_training_time)

        test_loss = evaluate(test_loader, mode="test")
        hold_loss = evaluate(hold_loader, mode="hold")
        hold_losses.append(hold_loss)
        test_losses.append(test_loss)

        stop_window.append(hold_loss)
        if len(stop_window) > num_stop_rounds:
            stop_window.pop(0)

        if hold_loss < best_hold:
            best_hold  = hold_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if len(stop_window) == num_stop_rounds and best_hold == min(stop_window):
            print("Early stop triggered.")
            break

    model.load_state_dict(best_state)
    model.eval()
    test_loss = evaluate(test_loader, mode="test")
    print(f"Final test_loss={test_loss:.4f}, hold_loss={hold_loss:.4f}")
    wandb.log({"final_test_loss": test_loss})

    if save_folder is not None:
        run_res = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'hold_losses': hold_losses,
            'total_training_time_list': total_training_time_list
        }
        torch.save(run_res, os.path.join(save_folder, f'{which}'))

    return model

def run_task_net_non_smooth(model,  layer, variables, params, X_train, Y_train, args):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    solver = layer

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for epoch in range(1000):
        opt.zero_grad()
        model.train()
        mu_pred_train, sig_pred_train = model(variables['X_train_'])
        train_loss = _dfl_loss(variables['Y_train_'], mu_pred_train, layer, params, constants.DEVICE)
        train_loss.backward()

        model.eval()
        mu_pred_test, sig_pred_test = model(variables['X_test_'])
        test_loss = _dfl_loss(variables['Y_test_'], mu_pred_test, layer, params, constants.DEVICE)

        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])
        hold_loss = _dfl_loss(variables['Y_hold_'], mu_pred_hold, layer, params, constants.DEVICE)
        print(f"Epoch = {epoch} | hold-out DFL = {hold_loss.item(): .4f} | test DFL = {test_loss.item(): .4f}")

        opt.step()

        # Early stopping
        hold_costs.append(hold_loss.sum().item())
        model_states.append(model.state_dict().copy())
        if epoch > 0 and epoch % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()
                best_model = model_classes.Net(
                    X_train[:,:-1], Y_train, [200, 200])
                best_model.load_state_dict(model_states[idx])
                if constants.USE_GPU:
                    best_model = best_model.to(constants.DEVICE)
                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model

def _dfl_loss(z_star, y_true, params):
    loss = task_loss(z_star, y_true, params).sum()
    return loss

def _dfl_loss_no_mean(z_star, y_true, params):
    loss = task_loss_no_mean(z_star, y_true, params) # (batch, 24)
    return loss

def print_graph(grad_fn):
    if grad_fn is None:
        return
    print(grad_fn)
    for next_fn, _ in grad_fn.next_functions:
        if next_fn is not None:
            print_graph(next_fn)

def compute_optimal_z_star(data_loader, params, model_net=None):
    solver = cvxpy_ps_parallel(params, mc_samples=1)
    z_star_list = []
    with torch.no_grad():
        for _, (x, y) in enumerate(data_loader):
            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            if model_net is None:
                z_star_tuple, info = solver(y)
            else:
                if isinstance(model_net, Diffsion):
                    y_preds = model_net.sample_elbo(x, test_mode=True)
                elif isinstance(model_net, model_classes.GaussianMLP):
                    mu, logvar = model_net(x)
                    sigma = torch.exp(0.5 * logvar)
                    y_preds = mu + sigma * torch.randn_like(mu)
                else:
                    y_preds = model_net(x)
                z_star_tuple, info = solver(y_preds)
            z_star = z_star_tuple[0]
            z_star_list.append(z_star)
            print(f"Test DFL = {_dfl_loss(z_star, y, params).mean().item():.4f}")
            print(f"Test RMSE = {rmse_loss(z_star, y).mean().item():.4f}")
    z_star_list = torch.cat(z_star_list, dim=0)
    return z_star_list

def compute_init_loss(layer, data_loader, params):
    with torch.no_grad():
        total_loss = 0
        layer.diffusion.model_net.train()
        for batch in data_loader:
            if len(batch) == 2:
                x, y = batch
                idx = None
            elif len(batch) == 3:
                x, y, idx = batch

            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            if idx is not None:
                z_star = layer(x, idx=idx, epoch=-1)
            else:
                z_star = layer(x)
            loss = _dfl_loss(z_star, y, params)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def run_weighted_rmse_net_mlp(
    model, 
    train_loader, 
    hold_loader, 
    test_loader, 
    params,
    args
):
    patience = 50

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0

    device = next(model.parameters()).device
    weights = torch.ones(params["n"], device=device)

    for epoch in range(1000):
        model.train()
        train_loss_epoch = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            mu_pred, sig_pred = model(x)

            Y_sched = model_classes.SolveScheduling(params)(
                mu_pred.double(), sig_pred.double()
            ).float()

            w = weights.unsqueeze(0).expand_as(y)
            weighted_rmse = (w * (Y_sched - y) ** 2).mean(dim=0).sqrt().sum()

            weighted_rmse.backward()
            optimizer.step()

            train_df_loss = _dfl_loss(Y_sched, y, params)
            train_loss_epoch += train_df_loss.item()

            print(f"Epoch {epoch}: train_df_loss={train_df_loss.item():.4f}")
            wandb.log({"train_df_loss": train_df_loss.item()})

        train_loss_epoch /= len(train_loader)
        
        model.eval()
        val_loss = 0
        hold_losses = []
        with torch.no_grad():
            for xh, yh in hold_loader:
                xh, yh = xh.to(device), yh.to(device)
                mu_h, sig_h = model(xh)
                Yh_sched = model_classes.SolveScheduling(params)(
                    mu_h.double(), sig_h.double()
                ).float()
                per_t_loss = _dfl_loss(Yh_sched, yh, params).item()  # shape (batch, T)
                val_loss += per_t_loss
                wandb.log({"hold_df_loss": per_t_loss})
                print(f"Epoch {epoch}: hold_loss={per_t_loss:.4f}")
                per_t = task_loss_no_mean(Yh_sched, yh, params)  
                hold_losses.append(per_t)
        hold_losses = torch.cat(hold_losses, dim=0)
        weights = hold_losses.mean(dim=0).detach()
        weights = weights / (weights.sum() / params["n"])
        weights = weights.to(device)

        val_loss /= len(hold_loader)
        print(f"Epoch {epoch}: final hold_df_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        test_df_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                mu_t, sig_t = model(x)
                Yt_sched = model_classes.SolveScheduling(params)(
                    mu_t.double(), sig_t.double()
                ).float()
                _test_loss = _dfl_loss(Yt_sched, y, params).item()
                test_df_loss += _test_loss
                print(f"Epoch {epoch}: test_df_loss={_test_loss:.4f}")
                wandb.log({"test_df_loss": _test_loss})
        test_df_loss /= len(test_loader)
        print(f"Epoch {epoch}: final test_df_loss={test_df_loss:.4f}")

    model.load_state_dict(best_state)

    model.eval()
    test_losses = []
    with torch.no_grad():
        for xt, yt in test_loader:
            xt, yt = xt.to(device), yt.to(device)
            mu_t, sig_t = model(xt)
            Yt_sched = model_classes.SolveScheduling(params)(
                mu_t.double(), sig_t.double()
            ).float()
            test_losses.append(task_loss(Yt_sched, yt, params).sum().item())
    avg_test_loss = sum(test_losses) / len(test_loader)
    print(f"Final test DFL loss: {avg_test_loss:.4f}")

    return model


def run_task_net_mlp(mlp_model, train_loader, hold_loader, test_loader, layer, params, args, which, save_folder):
    if which == "task_net_deterministic_mlp":
        opt = optim.Adam(mlp_model.parameters(), lr=args.lr)
    else:
        opt = optim.Adam(mlp_model.parameters(), lr=args.lr)

    # z_star_test = compute_optimal_z_star(test_loader, params)

    best_val = float('inf')
    patience = 100
    epochs_since = 0
    best_state = None

    cnt = 0
    total_training_time = 0.0
    train_dfl_losses = []
    val_dfl_losses = []
    test_dfl_losses = []
    total_training_time_list = []
    
    epoches = 500 if args.pretrain_epochs == 0 else 50
    for epoch in range(epoches):
        mlp_model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            opt.zero_grad()
            z_star = layer(x)
            loss = _dfl_loss(z_star, y, params)
            train_loss += loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), float('inf'))
            print(f"Epoch = {epoch}, train_df_loss = {loss.item(): .4f}, grad_norm = {grad_norm.item(): .4f}")
            wandb.log({"train_df_loss": loss.item(), "grad_norm": grad_norm.item()})
            opt.step()
            cnt += 1

        train_loss /= len(train_loader)
        total_training_time += time.time() - epoch_start_time

        mlp_model.eval()
        val_loss = 0.0
        num_samples = 0 
        with torch.no_grad():
            for x, y in hold_loader:
                x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                z_star = layer(x)
                loss = _dfl_loss_no_mean(z_star, y, params)
                per_sample = loss.sum(dim=1)
                val_loss += per_sample.sum().item()
                num_samples += y.shape[0]
                print(f"Epoch = {epoch} | hold-out DFL = {per_sample.mean().item():.4f}")
        val_loss /= num_samples
        print(f"Epoch = {epoch} | final hold-out DFL = {val_loss:.4f}\n")
        wandb.log({"hold_out_df_loss": val_loss})

        mlp_model.eval()
        test_loss = 0
        num_samples = 0 
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                z_star = layer(x)
                loss = _dfl_loss_no_mean(z_star, y, params)
                per_sample = loss.sum(dim=1)
                test_loss += per_sample.sum().item()
                num_samples += y.shape[0]
                print(f"Epoch = {epoch}, test_df_loss = {per_sample.mean().item(): .4f}")
            test_loss /= num_samples
            print(f"\nEpoch = {epoch}, final test_df_loss = {test_loss: .4f}\n")
            wandb.log({"test_df_loss": test_loss})

        train_dfl_losses.append(train_loss)
        val_dfl_losses.append(val_loss)
        test_dfl_losses.append(test_loss)
        total_training_time_list.append(total_training_time)

        # -------- early-stopping -----
        min_delta = 1e-4
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(mlp_model.state_dict())
            epochs_since = 0
        else:
            epochs_since += 1
            if epochs_since >= patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs).")
                break
        
    mlp_model.load_state_dict(best_state)

    test_loss = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            z_star = layer(x)
            test_loss += _dfl_loss(z_star, y, params).mean().item()
        test_loss /= len(test_loader)
        print(f"\nEpoch = {epoch}, test_df_loss = {test_loss: .4f}\n")
        wandb.log({"final_test_df_loss": test_loss})

    run_res = {
        'train_dfl_losses': train_dfl_losses,
        'val_dfl_losses': val_dfl_losses,
        'test_dfl_losses': test_dfl_losses,
        'total_training_time_list': total_training_time_list
    }
    torch.save(run_res, os.path.join(save_folder, f'{which}'))
    torch.save(mlp_model.state_dict(), os.path.join(save_folder, f'{which}.pth'))

    return mlp_model

def run_two_stage(model, train_loader, hold_loader, test_loader, variables, params):
    z_star_train = compute_optimal_z_star(train_loader, params, model_net=model)
    z_star_hold = compute_optimal_z_star(hold_loader, params, model_net=model)
    z_star_test = compute_optimal_z_star(test_loader, params, model_net=model)

    train_loss_task = task_loss(
            z_star_train, variables['Y_train_'], params)
    
    hold_loss_task = task_loss(
            z_star_hold, variables['Y_hold_'], params)
    
    test_loss_task = task_loss(
            z_star_test, variables['Y_test_'], params)

    print(f"Train DFL = {train_loss_task.mean().item():.4f}")
    print(f"Hold DFL = {hold_loss_task.mean().item():.4f}")
    print(f"Test DFL = {test_loss_task.mean().item():.4f}")

    return model

def get_grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    grads = [g for g in grads if g is not None]
    return torch.sqrt(sum(g.pow(2).sum() for g in grads) + 1e-12)

def _named_grads(loss: torch.Tensor, module: torch.nn.Module, retain_graph: bool=False):
    items  = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    params = [p for _, p in items]
    grads  = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                 create_graph=False, allow_unused=True)
    out = OrderedDict()
    for (name, p), g in zip(items, grads):
        out[name] = (torch.zeros_like(p) if g is None else g.detach())
    return out

def _pack_vec(ng: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in ng.values()]) if len(ng) else torch.tensor([])

def cos_by_param_from_grads(ga: OrderedDict, gb: OrderedDict):
    per_param = {name: _cos(ga[name], gb[name]) for name in ga.keys()}
    overall   = _cos(_pack_vec(ga), _pack_vec(gb))
    return overall, per_param

def _cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.cosine_similarity(u.reshape(1, -1),
                              v.reshape(1, -1),
                              dim=1, eps=1e-12).squeeze(0)

# differentiable optimization with diffusion probability model
def run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, layer, params, args, which, save_folder, other_layers=None):
    opt = optim.Adam(diffusion.model_net.parameters(), lr=args.lr)
    # opt = optim.AdamW(diffusion.model_net.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float('inf')
    patience = 500
    epochs_since = 0
    best_state = None
    train_dfl_losses = []
    val_dfl_losses = []
    test_dfl_losses = []
    total_training_time_list = []

    cnt = 0
    interval = args.interval
    total_training_time = 0.0
            
    # init_train_loss = compute_init_loss(layer, train_loader, params)
    # train_dfl_losses.append(init_train_loss)
    # print(f"Init train DFL = {init_train_loss:.4f}")
    # init_val_loss = compute_init_loss(layer, hold_loader, params)
    # val_dfl_losses.append(init_val_loss)
    # print(f"Init hold DFL = {init_val_loss:.4f}")
    # init_test_loss = compute_init_loss(layer, test_loader, params)
    # test_dfl_losses.append(init_test_loss)
    # print(f"Init test DFL = {init_test_loss:.4f}")

    epoches = 500 if args.pretrain_epochs == 0 else 50

    for epoch in range(int(args.interval) * epoches):
        torch.cuda.reset_peak_memory_stats(device=args.cuda_device)
        epoch_start_time = time.time()
        diffusion.model_net.train()
        epoch_loss = 0
        num_samples = 0
        layer.interval = interval

        # for fold, (train_loader, hold_loader) in enumerate(fold_loaders):
        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                idx = None
            elif len(batch) == 3:
                x, y, idx = batch

            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            opt.zero_grad(set_to_none=True)
            if other_layers is not None:
                with torch.enable_grad():
                    _x = x.repeat(args.mc_samples, 1) # (batch_size * mc_samples * 5, 24)
                    layer_reparam = other_layers['layer_reparam']
                    y_preds = layer_reparam.diffusion.sample_elbo(_x, test_mode=False)

                z_star_reparam = layer_reparam(x, _y=y_preds)
                loss_reparam = task_loss(z_star_reparam, y, params).sum()

                layer_resample = other_layers['layer_resample']
                z_star_resample = layer_resample(x, _y=y_preds.clone().detach())
                loss_resample = task_loss(z_star_resample, y, params).sum()

                layer_distr_10 = other_layers['layer_distr_10']
                layer_distr_50 = other_layers['layer_distr_50']
                layer_distr_100 = other_layers['layer_distr_100']
                layer_distr_500 = other_layers['layer_distr_500']
                z_star_distr_10 = layer_distr_10(x, _y=y_preds.clone().detach())
                loss_distr_10 = task_loss(z_star_distr_10, y, params).sum()
                z_star_distr_50 = layer_distr_50(x, _y=y_preds.clone().detach())
                loss_distr_50 = task_loss(z_star_distr_50, y, params).sum()
                z_star_distr_100 = layer_distr_100(x, _y=y_preds.clone().detach())
                loss_distr_100 = task_loss(z_star_distr_100, y, params).sum()
                z_star_distr_500 = layer_distr_500(x, _y=y_preds.clone().detach())
                loss_distr_500 = task_loss(z_star_distr_500, y, params).sum()

                g_reparam  = _named_grads(loss_reparam,  layer_reparam.diffusion.model_net, retain_graph=True)
                g_distr_10    = _named_grads(loss_distr_10,    layer_distr_10.diffusion.model_net, retain_graph=True)
                g_distr_50    = _named_grads(loss_distr_50,    layer_distr_50.diffusion.model_net, retain_graph=True)
                g_distr_100    = _named_grads(loss_distr_100,    layer_distr_100.diffusion.model_net, retain_graph=True)
                g_distr_500    = _named_grads(loss_distr_500,    layer_distr_500.diffusion.model_net, retain_graph=True)
                g_resample = _named_grads(loss_resample, layer_resample.diffusion.model_net, retain_graph=True)

                overall_dp_10, per_dp_10 = cos_by_param_from_grads(g_distr_10, g_reparam)
                overall_dp_50, per_dp_50 = cos_by_param_from_grads(g_distr_50, g_reparam)
                overall_dp_100, per_dp_100 = cos_by_param_from_grads(g_distr_100, g_reparam)
                overall_dp_500, per_dp_500 = cos_by_param_from_grads(g_distr_500, g_reparam)
                overall_rp, per_rp = cos_by_param_from_grads(g_resample, g_reparam)

                print(f"cos_dp_10 = {overall_dp_10.item():.4f}, cos_dp_50 = {overall_dp_50.item():.4f}, cos_dp_100 = {overall_dp_100.item():.4f}, cos_dp_500 = {overall_dp_500.item():.4f}, cos_rp = {overall_rp.item():.4f}")

                opt.zero_grad(set_to_none=True)

                diffusion_loss = 0.0
                num_samples = 10
                for _ in range(num_samples):
                    t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=constants.DEVICE).long()
                    diffusion_loss += diffusion.diffusion_loss(y, t, x)
                diffusion_loss /= num_samples

                z_star_reparam = layer_reparam(x, _y=y_preds)
                dfl_loss = task_loss_no_mean(z_star_reparam, y, params)
            else:
                if idx is not None:
                    # z_star = layer(x, y, idx=idx, epoch=epoch)
                    z_star = layer(x, idx=idx, epoch=epoch)
                else:
                    z_star = layer(x)

                diffusion_loss = 0.0
                for _ in range(10):
                    cur_batch_size = x.shape[0]
                    t = torch.randint(0, diffusion.timesteps, (cur_batch_size,), device=constants.DEVICE).long()
                    diffusion_loss += diffusion.diffusion_loss(y, t, x)
                diffusion_loss /= 10.0
                dfl_loss = task_loss_no_mean(z_star, y, params)

            if args.pretrain_epochs == 0:
                per_sample = dfl_loss.sum(dim=1)
                dfl_grad_norm = get_grad_norm(dfl_loss.mean(), diffusion.model_net.parameters())
                mse_grad_norm = get_grad_norm(diffusion_loss, diffusion.model_net.parameters())
                print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
                alpha = 0.2
                # loss = alpha * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - alpha) * diffusion_loss
                loss = alpha * dfl_loss.mean() + (1 - alpha) * diffusion_loss * (dfl_grad_norm / mse_grad_norm + 1e-8).detach()
            else:
                per_sample = dfl_loss.sum(dim=1)
                dfl_grad_norm = get_grad_norm(dfl_loss.mean(), diffusion.model_net.parameters())
                mse_grad_norm = get_grad_norm(diffusion_loss, diffusion.model_net.parameters())
                print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
                alpha = 0.9
                loss = alpha * dfl_loss.mean() + (1 - alpha) * diffusion_loss * (dfl_grad_norm / mse_grad_norm + 1e-8).detach()
                # loss = dfl_loss.mean()

            torch.cuda.synchronize(args.cuda_device)
            fwd_peak = torch.cuda.max_memory_allocated(args.cuda_device) / 1024**3  # GB

            torch.cuda.reset_peak_memory_stats(device=args.cuda_device)

            num_samples += y.shape[0] # batch size

            time_start = time.time()
            loss.backward()

             # save first graident
            # if epoch == 0:
            #     first_grad = [p.grad for p in diffusion.model_net.parameters() if p.requires_grad]
            #     torch.save(first_grad, os.path.join(save_folder, f'{which}_first_grad.pth'))
            #     exit()
                
            time_end = time.time()
            torch.cuda.synchronize(args.cuda_device)
            bwd_peak = torch.cuda.max_memory_allocated(args.cuda_device) / 1024**3  # GB
            # print(f"Time taken for backward: {time_end - time_start} seconds")
            # print(f"Forward peak: {fwd_peak:.4f} GB, backward peak: {bwd_peak:.4f} GB")
            
            grad_norm = torch.nn.utils.clip_grad_norm_(diffusion.model_net.parameters(), float('inf'))

            print(f"Epoch = {epoch}, train_df_loss = {dfl_loss.sum(dim=-1).mean(): .4f}, grad_norm = {grad_norm.item(): .4f}")

            wandb.log({"train_df_loss": dfl_loss.sum(dim=-1).mean(), "grad_norm": grad_norm.item()})
            opt.step()
            epoch_loss += dfl_loss.sum(dim=-1).mean().item()
            cnt += 1

            # del loss, z_star
            torch.cuda.empty_cache()

        epoch_loss /= len(train_loader)
        total_training_time = time.time() - epoch_start_time

        if args.pretrain_epochs > 0 and epoch > 5 and epoch_loss > 5:
            print(f"Early stop at epoch {epoch}, train_df_loss = {epoch_loss:.4f}")
            break
        
        diffusion.model_net.eval()
        val_loss = 0.0
        num_samples = 0 
        layer.interval = 1
        with torch.no_grad():
            for batch in hold_loader:
                if len(batch) == 2:
                    x, y = batch
                    idx = None
                elif len(batch) == 3:
                    x, y, idx = batch

                x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                if idx is not None:
                    # z_star = layer(x, y, idx=idx, epoch=-1)
                    z_star = layer(x, idx=idx, epoch=-1)
                else:
                    z_star = layer(x)
                loss = _dfl_loss_no_mean(z_star, y, params)
                per_sample = loss.sum(dim=1)
                val_loss += per_sample.sum().item()
                num_samples += y.shape[0]
                print(f"Epoch = {epoch} | hold-out DFL = {per_sample.mean().item():.4f}")
        val_loss /= num_samples
        print(f"Epoch = {epoch} | final hold-out DFL = {val_loss:.4f}\n")

        test_loss = 0
        num_samples = 0 
        with torch.no_grad():
            layer.interval = 1
            for batch in test_loader:
                if len(batch) == 2:
                    x, y = batch
                    idx = None
                elif len(batch) == 3:
                    x, y, idx = batch

                x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                if idx is not None:
                    # z_star = layer(x, y, idx=idx, epoch=-1)
                    z_star = layer(x, idx=idx, epoch=-1)
                else:
                    z_star = layer(x)
                loss = _dfl_loss_no_mean(z_star, y, params)
                per_sample = loss.sum(dim=1)
                test_loss += per_sample.sum().item()
                num_samples += y.shape[0]
                print(f"Epoch = {epoch}, test_df_loss = {per_sample.mean().item(): .4f}")
            test_loss /= num_samples
            print(f"\nEpoch = {epoch}, final test_df_loss = {test_loss: .4f}, total_training_time = {total_training_time:.4f}\n")
            wandb.log({"train_df_loss_epoch": epoch_loss, "total_training_time": total_training_time, "hold_out_df_loss": val_loss, "test_df_loss": test_loss})

            train_dfl_losses.append(epoch_loss)
            val_dfl_losses.append(val_loss)
            test_dfl_losses.append(test_loss)
            total_training_time_list.append(total_training_time)

            # -------- early-stopping -----
            min_delta = 1e-4
            if val_loss < best_val - min_delta:
                best_val     = val_loss
                best_state   = copy.deepcopy(diffusion.model_net.state_dict())
                epochs_since = 0
            else:
                epochs_since += 1
                if epochs_since >= patience:
                    print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs).")
                    break

    diffusion.model_net.load_state_dict(best_state)

    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x, y = batch
                idx = None
            elif len(batch) == 3:
                x, y, idx = batch
                
            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            if idx is not None:
                z_star = layer(x, idx=idx, epoch=-1)
            else:
                z_star = layer(x)
            test_loss += _dfl_loss(z_star, y, params).mean().item()
        test_loss /= len(test_loader)
        print(f"\nEpoch = {epoch}, test_df_loss = {test_loss: .4f}\n")
        wandb.log({"final_test_df_loss": test_loss})

    run_res = {
        'train_dfl_losses': train_dfl_losses,
        'val_dfl_losses': val_dfl_losses,
        'test_dfl_losses': test_dfl_losses,
        'total_training_time_list': total_training_time_list
    }
    torch.save(run_res, os.path.join(save_folder, f'{which}'))
    
    return diffusion

def run_task_net_diffusion_with_time(variables, params, args):
    def train_one_epoch(layer, loader, params):
        try:
            layer.train()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            opt = optim.Adam(layer.diffusion.model_net.parameters(), lr=1e-4)

            for _ in range(5):
                for batch in loader: 
                    if len(batch) == 2:
                        x, y = batch
                        idx = None
                    elif len(batch) == 3:
                        x, y, idx = batch

                    x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                    if idx is not None: 
                        z_star = layer(x, idx=idx, epoch=1)
                    else:
                        z_star = layer(x)
                    opt.zero_grad()
                    loss = _dfl_loss(z_star, y, params)
                    loss.backward()
                    opt.step()
            
            torch.cuda.synchronize()           

            epoch_t = time.perf_counter() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            return epoch_t, peak_gb
        except RuntimeError as e:
            print(f"Out of memory")
            traceback.print_exc(limit=1)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            gc.collect()
            return None, None
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc(limit=1)
            return None, None
    
    def build_layer(method, diffusion, params, mc):
        if method == "diffusion_point_est":
            return DiffusionCvxpyModule_reparam(diffusion, params, mc, distr_est=False)
        if method == "diffusion_distr_est":
            return DiffusionCvxpyModule_distr(diffusion, params, mc, distr_est=True)
        if method == "diffusion_replay":
            return DiffusionCvxpyModuleReplay(diffusion, params, mc, interval=args.interval)
        raise ValueError(method)
    
    mc_samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    methods = ["diffusion_replay", "diffusion_point_est", "diffusion_distr_est"]

    X_train2_ = variables['X_train_']
    Y_train2_ = variables['Y_train_']
    base_save = 'YOUR_SAVE_FOLDER'
    save_folder = os.path.join(base_save, 'diffusion_with_time')

    for mc in mc_samples:
        fails = 0
        for method in methods:
            fixed_batch_size = 64
            if method == "diffusion_replay":
                train_ds = IndexedDataset(variables['X_train_'], variables['Y_train_'])
                hold_ds = IndexedDataset(variables['X_hold_'], variables['Y_hold_'])
            else:
                train_ds = torch.utils.data.TensorDataset(variables['X_train_'], variables['Y_train_'])
                hold_ds = torch.utils.data.TensorDataset(variables['X_hold_'], variables['Y_hold_'])
                
            loader = torch.utils.data.DataLoader(train_ds, batch_size=fixed_batch_size, shuffle=True)
            hold_loader = torch.utils.data.DataLoader(hold_ds, batch_size=fixed_batch_size, shuffle=True)
            print(f"Start method = {method}, sample size = {mc}")
            diffusion = Diffsion(x_dim=X_train2_.shape[1],
                                    y_dim=Y_train2_.shape[1],
                                    timesteps=1000,
                                    device=constants.DEVICE)
            
            diffusion.pretrain_diffusion(variables['X_train_'], variables['Y_train_'], hold_loader, base_save, args)

            layer = build_layer(method, diffusion, params, mc)
            epoch_t, peak_gb = train_one_epoch(layer, loader, params)
            if epoch_t is None:
                result = {
                    "method"     : method,
                    "batch_size" : fixed_batch_size,
                    "mc_samples" : mc,
                    "status"     : "OOM",
                }
                wandb.log(result)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                with open(os.path.join(save_folder, f'{method}_mc_{mc}_bs_{fixed_batch_size}_OOM.pkl'), 'wb') as f:
                    pickle.dump(result, f)
                print(f"[{method:9s}] bs={fixed_batch_size:3d} mc={mc:2d} OOM")
                del layer, diffusion
                fails += 1
                if fails == 3:
                    break
                else:
                    continue

            result = {
                "epoch_time_s" : epoch_t,
                "peak_mem_gb"  : peak_gb,
                "method"       : method,
                "batch_size"   : fixed_batch_size,
                "mc_samples"   : mc,
                "status"       : "success",
            }
            wandb.log(result)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            with open(os.path.join(save_folder, f'{method}_mc_{mc}_bs_{fixed_batch_size}_success.pkl'), 'wb') as f:
                pickle.dump(result, f)
            print(f"[{method:9s}] bs={fixed_batch_size:3d} mc={mc:2d} "
                    f"time={epoch_t:6.2f}s  mem={peak_gb:4.2f} GB")
            del layer, diffusion


    for bs in batch_sizes:
        fails = 0
        for method in methods:
            if method == "diffusion_replay":
                train_ds = IndexedDataset(variables['X_train_'], variables['Y_train_'])
                hold_ds = IndexedDataset(variables['X_hold_'], variables['Y_hold_'])
            else:
                train_ds = torch.utils.data.TensorDataset(variables['X_train_'], variables['Y_train_'])
                hold_ds = torch.utils.data.TensorDataset(variables['X_hold_'], variables['Y_hold_'])

            loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
            hold_loader = torch.utils.data.DataLoader(hold_ds, batch_size=bs, shuffle=True)
            print(f"Start method = {method}, batch size = {bs}")

            fixed_mc_sample = 10
            diffusion = Diffsion(x_dim=X_train2_.shape[1],
                                y_dim=Y_train2_.shape[1],
                                timesteps=1000,
                                device=constants.DEVICE)
            diffusion.pretrain_diffusion(variables['X_train_'], variables['Y_train_'], hold_loader, base_save, args)

            layer = build_layer(method, diffusion, params, fixed_mc_sample)
            epoch_t, peak_gb = train_one_epoch(layer, loader, params)
            if epoch_t is None:
                result = {
                    "method"     : method,
                    "batch_size" : bs,
                    "mc_samples" : fixed_mc_sample,
                    "status"     : "OOM",
                }
                wandb.log(result)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                with open(os.path.join(save_folder, f'{method}_bs_{bs}_mc_{fixed_mc_sample}_OOM.pkl'), 'wb') as f:
                    pickle.dump(result, f)
                print(f"[{method:9s}] bs={bs:3d} mc={fixed_mc_sample:2d} OOM")
                del layer, diffusion
                fails += 1
                if fails == 3:
                    break
                else:
                    continue
            result = {
                "epoch_time_s" : epoch_t,
                "peak_mem_gb"  : peak_gb,
                "method"       : method,
                "batch_size"   : bs,
                "mc_samples"   : fixed_mc_sample,
                "status"       : "success",
            }
            wandb.log(result)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            with open(os.path.join(save_folder, f'{method}_bs_{bs}_mc_{fixed_mc_sample}_success.pkl'), 'wb') as f:
                pickle.dump(result, f)
            print(f"[{method:9s}] bs={bs:3d} mc={fixed_mc_sample:2d} "
                    f"time={epoch_t:6.2f}s  mem={peak_gb:4.2f} GB")
            del layer, diffusion

def eval_diffusion(which, diffusion, variables, params, mc_samples, save_folder):
    layer = cvxpy_ps_parallel(params, mc_samples)
    with torch.no_grad():
        diffusion.model_net.eval()

        y_pred_train = diffusion.sample_elbo(variables['X_train_'].repeat(mc_samples, 1), test_mode=True)
        y_pred_test = diffusion.sample_elbo(variables['X_test_'].repeat(mc_samples, 1), test_mode=True)
        y_pred_hold = diffusion.sample_elbo(variables['X_hold_'].repeat(mc_samples, 1), test_mode=True)
        y_pred_train = y_pred_train.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()
        y_pred_test = y_pred_test.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()
        y_pred_hold = y_pred_hold.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

        train_rmse = rmse_loss(y_pred_train, variables['Y_train_'].repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
        test_rmse = rmse_loss(y_pred_test, variables['Y_test_'].repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
        hold_rmse = rmse_loss(y_pred_hold, variables['Y_hold_'].repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())

        with open(
            os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
            np.save(f, train_rmse)
        with open(
            os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
            np.save(f, test_rmse)
        with open(
            os.path.join(save_folder, '{}_hold_rmse'.format(which)), 'wb') as f:
            np.save(f, hold_rmse)
        
        z_star_train, _ = layer(y_pred_train.double())
        z_star_train = z_star_train[0]
        train_loss_task = task_loss(
            z_star_train, variables['Y_train_'], params)
        
        z_star_test, _ = layer(y_pred_test.double())
        z_star_test = z_star_test[0]
        test_loss_task = task_loss(
            z_star_test, variables['Y_test_'], params)

        z_star_hold, _ = layer(y_pred_hold.double())
        z_star_hold = z_star_hold[0]
        hold_loss_task = task_loss(
            z_star_hold, variables['Y_hold_'], params)

        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))
        torch.save(hold_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_hold_task'.format(which)))
        
        print(f"test_loss_task = {test_loss_task.sum().item():.4f}")

def eval_net(which, model, variables, params, save_folder, mc_samples=None):
    if isinstance(model, model_classes.GaussianMLP):
        solver = cvxpy_ps_parallel(params, mc_samples)
        model.eval()
        mu_pred_train, logvar_pred_train = model(variables['X_train_'].repeat(mc_samples, 1))
        mu_pred_test, logvar_pred_test = model(variables['X_test_'].repeat(mc_samples, 1))
        sig_pred_train = torch.exp(0.5 * logvar_pred_train)
        sig_pred_test = torch.exp(0.5 * logvar_pred_test)
        y_preds_train = mu_pred_train + sig_pred_train * torch.randn_like(mu_pred_train)
        y_preds_test = mu_pred_test + sig_pred_test * torch.randn_like(mu_pred_test)

        y_preds_train = y_preds_train.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()
        y_preds_test = y_preds_test.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

        # Eval model on rmse
        train_rmse = rmse_loss(y_preds_train, variables['Y_train_'].repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
        test_rmse = rmse_loss(y_preds_test, variables['Y_test_'].repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())

        with open(
            os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
            np.save(f, train_rmse)

        with open(
            os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
            np.save(f, test_rmse)

        # Eval model on task loss
        Y_sched_train, _ = solver(y_preds_train.double())
        Y_sched_train = Y_sched_train[0]
        train_loss_task = task_loss(
            Y_sched_train.float(), variables['Y_train_'], params)

        Y_sched_test, _ = solver(y_preds_test.double())
        Y_sched_test = Y_sched_test[0]
        test_loss_task = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)

        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))

        print(f"test_loss_task = {test_loss_task.sum().item():.4f}")
    elif isinstance(model, model_classes.MLP):
        solver = cvxpy_ps_parallel(params, 1)
        model.eval()
        y_preds_train = model(variables['X_train_'])
        y_preds_test = model(variables['X_test_'])
        
        train_rmse = rmse_loss(y_preds_train, variables['Y_train_'])
        test_rmse = rmse_loss(y_preds_test, variables['Y_test_'])

        with open(
            os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
            np.save(f, train_rmse)

        with open(
            os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
            np.save(f, test_rmse)
        
        Y_sched_train, _ = solver(y_preds_train.double().unsqueeze(1))
        Y_sched_train = Y_sched_train[0]
        train_loss_task = task_loss(
            Y_sched_train.float(), variables['Y_train_'], params)
        
        Y_sched_test, _ = solver(y_preds_test.double().unsqueeze(1))
        Y_sched_test = Y_sched_test[0]
        test_loss_task = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)
        
        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))

        print(f"test_loss_task = {test_loss_task.sum().item():.4f}")

if __name__ == "__main__":
    from main import load_data_with_features

    device = "cuda:7"

    seed = 2

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    X1, Y1 = load_data_with_features(os.path.join(dataset_dir, 'pjm_load_data_2008-11.txt'))
    X2, Y2 = load_data_with_features(os.path.join(dataset_dir, 'pjm_load_data_2012-16.txt'))
    X = np.concatenate((X1, X2), axis=0)
    X[:,:-1] = \
        (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)
    
    Y = np.concatenate((Y1, Y2), axis=0)

    # Train, test split.
    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]

    # Construct tensors (without intercepts).
    X_train_ = torch.tensor(X_train[:,:-1], dtype=torch.float, device=device)
    Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=device)
    X_test_ = torch.tensor(X_test[:,:-1], dtype=torch.float, device=device)
    Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=device)
    
    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}

    base_save = 'YOUR_SAVE_FOLDER'
    # runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    runs = [0, 1]
    
    with torch.no_grad():
        for run in runs:
            random.seed(run + seed)
            np.random.seed(run + seed)
            torch.manual_seed(run + seed)
            torch.cuda.manual_seed_all(run + seed)
            torch.backends.cudnn.deterministic = True

            save_folder = os.path.join(base_save, str(run))

            th_frac = 0.8
            inds = np.random.permutation(X_train.shape[0])
            train_inds = inds[ :int(X_train.shape[0] * th_frac)]
            hold_inds = inds[int(X_train.shape[0] * th_frac):]
            X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
            Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]
            X_train2_ = torch.tensor(X_train2[:,:-1], dtype=torch.float32, device=device)
            Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=device)
            X_hold2_ = torch.tensor(X_hold2[:,:-1], dtype=torch.float32, device=device)
            Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=device)
            variables = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
                        'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
                        'X_test_': X_test_, 'Y_test_': Y_test_}
            
            # diffusion_timesteps = 300
            # diffusion_timesteps = 1000
            # diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=device)

            # which = "diffusion_distr_est_resample"
            # diffusion.model_net.load_state_dict(torch.load(save_folder + f'/{which}.pth'))
            # diffusion.model_net.eval()
            # eval_diffusion(which, diffusion, variables, params, 40, save_folder)

            # which = "diffusion_point_est"
            # diffusion.model_net.load_state_dict(torch.load(save_folder + f'/{which}.pth'))
            # diffusion.model_net.eval()
            # eval_diffusion(which, diffusion, variables, params, 40, save_folder)

            model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1]).to(device)

            # save_folder = os.path.join(base_save, str(run))
            # which = "task_net_mlp_gaussian_reparam"
            # model_task.load_state_dict(torch.load(save_folder + f'/{which}.pth'))
            # eval_net(which, model_task, variables, params, save_folder)

            which = "task_net_mlp_gaussian_distr"
            model_task.load_state_dict(torch.load(save_folder + f'/{which}.pth'))
            eval_net(which, model_task, variables, params, save_folder, mc_samples=40)

            print(f"End of run {run}")