#!/usr/bin/env python3

import os
import numpy as np
import operator
from functools import reduce
import sys
sys.path.append('.')
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import wandb

from qpth.qp import QPFunction

import math
from collections import OrderedDict
import constants
import model_classes
from diffusion_opt import Diffsion
from cvxpy_toy_kkt import cvxpy_toy_parallel_kkt, task_loss, task_loss_no_mean


def _dfl_loss(z_star, y_true, params):
    loss = task_loss(z_star, y_true, params).sum()
    return loss

def _dfl_loss_no_mean(z_star, y_true, params):
    loss = task_loss_no_mean(z_star, y_true, params)
    return loss

def rmse_loss(mu_pred, Y_actual):
    if mu_pred.ndim == 3:
        return ((mu_pred - Y_actual)**2).mean().sqrt().data.cpu().numpy()
    else:
        # return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()
        return ((mu_pred - Y_actual)**2).mean().sqrt().data.cpu().numpy()

def rmse_loss_weighted(mu_pred, Y_actual, weights):
    return ((weights * (mu_pred - Y_actual)**2).mean(dim=0).sqrt()).sum()

def run_rmse_net(model, variables, set_sig=True, pretrain_epochs=1000):
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for i in range(pretrain_epochs):
        opt.zero_grad()
        if isinstance(model, model_classes.GaussianMLP):
            model.train()
            mu, logvar = model(variables['X_train_'])
            var = torch.exp(logvar)
            loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
            train_loss = loss_fn(mu, variables['Y_train_'], var)
        else:
            model.train()
            if set_sig:
                train_loss = nn.MSELoss()(
                    model(variables['X_train_'])[0], variables['Y_train_'])
            else:
                train_loss = nn.MSELoss()(
                    model(variables['X_train_']), variables['Y_train_'])
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            if isinstance(model, model_classes.GaussianMLP):
                mu, logvar = model(variables['X_test_'])
                var = torch.exp(logvar)
                loss_fn = nn.GaussianNLLLoss(full=True, reduction='mean')
                test_loss = loss_fn(mu, variables['Y_test_'], var)
            else:
                if set_sig:
                    test_loss = nn.MSELoss()(
                        model(variables['X_test_'])[0], variables['Y_test_'])
                else:
                    test_loss = nn.MSELoss()(
                        model(variables['X_test_']), variables['Y_test_'])
        if i % 200 == 0:
            print(i, train_loss.item(), test_loss.item())

    if set_sig and not isinstance(model, model_classes.GaussianMLP):
        model.eval()
        model.set_sig(variables['X_train_'], variables['Y_train_'])

    return model

def run_task_net_gaussian(mlp_model, train_loader, hold_loader, test_loader, layer, params, args, which, save_folder):
    if "task_net_mlp_gaussian_reparam" in which:
        opt = optim.Adam(mlp_model.parameters(), lr=args.lr)
    elif "task_net_mlp_gaussian_distr" in which:
        opt = optim.Adam(mlp_model.parameters(), lr=args.lr)
    elif "task_net_deterministic_mlp" in which:
        opt = optim.Adam(mlp_model.parameters(), lr=args.lr)

    # z_star_test = compute_optimal_z_star(test_loader, params)

    best_val = float('inf')
    patience = 50
    epochs_since = 0
    best_state = None

    train_dfl_losses = []
    val_dfl_losses = []
    test_dfl_losses = []
    total_training_time_list = []
    total_training_time = 0.0

    cnt = 0

    mse_fn = nn.MSELoss()

    epochs = 50 if args.pretrain_epochs == 0 else 20

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        mlp_model.train()
        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
            opt.zero_grad()

            y_preds, z_star = layer(x)
            dfl_loss = _dfl_loss(z_star, y, params)
            if args.mc_samples > 1:
                y_preds = y_preds.mean(dim=1)
            mse_loss = mse_fn(y_preds, y)
            dfl_grad_norm = get_grad_norm(dfl_loss, [p for p in mlp_model.parameters() if p.requires_grad])
            mse_grad_norm = get_grad_norm(mse_loss, [p for p in mlp_model.parameters() if p.requires_grad])
            # print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
            alpha = 0.9
            loss = alpha * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - alpha) * mse_loss

            epoch_loss += loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), float('inf'))
            print(f"Epoch = {epoch}, train_df_loss = {dfl_loss.mean().item(): .4f}, grad_norm = {grad_norm.item(): .4f}")

            wandb.log({"train_df_loss": dfl_loss.mean().item(), "grad_norm": grad_norm.item()})
            opt.step()
            cnt += 1
        time_end = time.time()
        total_training_time += time_end - epoch_start_time
        epoch_loss /= len(train_loader)

        mlp_model.eval()
        val_loss = 0.0
        num_samples = 0 
        with torch.no_grad():
            for x, y in hold_loader:
                x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                y_preds, z_star = layer(x)
                loss = _dfl_loss_no_mean(z_star, y, params)
                per_sample = loss.mean()
                val_loss += per_sample.sum().item()
                num_samples += y.shape[0]
                print(f"Epoch = {epoch} | hold-out DFL = {per_sample.mean().item():.4f}")
        val_loss /= len(hold_loader)
        print(f"Epoch = {epoch} | final hold-out DFL = {val_loss:.4f}\n")
        wandb.log({"hold_out_df_loss": val_loss})

        test_loss = 0
        num_samples = 0 
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                y_preds, z_star = layer(x)
                # unexpected_rows = ~ (torch.isclose(z_star.sum(dim=1), torch.tensor(1.0, device=z_star.device, dtype=z_star.dtype)) & torch.isclose(z_star.prod(dim=1), torch.tensor(0.0, device=z_star.device, dtype=z_star.dtype), atol=1e-7))
                # if torch.any(unexpected_rows) and 'deterministic' in which: 
                #     print(f"Unexpected z_star values at indices {unexpected_rows.nonzero(as_tuple=True)[0]}")
                loss = _dfl_loss_no_mean(z_star, y, params)
                per_sample = loss.mean()
                test_loss += per_sample.sum().item()
                num_samples += y.shape[0]
                print(f"Epoch = {epoch}, test_df_loss = {per_sample.mean().item(): .4f}")
            test_loss /= len(test_loader)
            print(f"\nEpoch = {epoch}, final test_df_loss = {test_loss: .4f}\n")
            wandb.log({"test_df_loss": test_loss})

        train_dfl_losses.append(epoch_loss)
        val_dfl_losses.append(val_loss)
        test_dfl_losses.append(test_loss)
        total_training_time_list.append(total_training_time)

        # -------- early-stopping -----
        min_delta = 1e-4
        if val_loss < best_val - min_delta:
            best_val     = val_loss
            best_state   = copy.deepcopy(mlp_model.state_dict())
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
            y_preds, z_star = layer(x)
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
    torch.save(run_res, os.path.join(save_folder, '{}_run_res'.format(which)))
    return mlp_model, run_res

def get_grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    grads = [g for g in grads if g is not None]
    return torch.sqrt(sum(g.pow(2).sum() for g in grads) + 1e-12)


# def _grad_vec_from_loss(loss: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
#     params = [p for p in module.parameters() if p.requires_grad]
#     grads  = torch.autograd.grad(
#         loss, params, retain_graph=False, create_graph=False, allow_unused=True
#     )
#     flat = []
#     for p, g in zip(params, grads):
#         if g is None:
#             flat.append(torch.zeros_like(p, device=p.device, dtype=p.dtype).reshape(-1))
#         else:
#             flat.append(g.detach().reshape(-1))
#     v = torch.cat(flat)
#     # return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
#     return v


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
    per_param_norms = {
        name: (torch.linalg.norm(ga[name]), torch.linalg.norm(gb[name]))
        for name in ga.keys()
    }
    overall = _cos(_pack_vec(ga), _pack_vec(gb))
    v = (_pack_vec(ga) - _pack_vec(gb))
    rms = torch.linalg.norm(v) / v.numel()**0.5
    return overall, rms
    # return overall, overall_norms, per_param, per_param_norms

def _cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.cosine_similarity(u.reshape(1, -1),
                              v.reshape(1, -1),
                              dim=1, eps=1e-12).squeeze(0)

# differentiable optimization with diffusion probability model
def run_task_net_diffusion(model, train_loader, hold_loader, test_loader, layer, params, args, which, save_folder, other_layers=None):
    # opt = optim.AdamW(diffusion.model_net.parameters(), lr=args.lr, weight_decay=1e-3)
    if "diffusion" in which:
        diffusion = model
        opt = optim.Adam(diffusion.model_net.parameters(), lr=args.lr)
    elif "cnf" in which:
        cnf = model
        opt = optim.Adam(cnf.flow.parameters(), lr=args.lr)

    best_val = float('inf')
    patience = 300
    epochs_since = 0
    best_state = None

    cnt = 0

    total_training_time = 0

    train_dfl_losses = []
    val_dfl_losses = []
    test_dfl_losses = []
    total_training_time_list = []
    epochs = 100 if args.pretrain_epochs == 0 else 10
    
    for epoch in range(epochs):
        time_start = time.time()
        if "diffusion" in which:
            diffusion.model_net.train()
            layer.diffusion.model_net.train()
        elif "cnf" in which:
            cnf.flow.train()
            layer.cnf.flow.train()
        
        epoch_loss = 0
        grad_cnt = 0
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
                loss_reparam = task_loss(z_star_reparam, y, params)
                # loss_reparam.backward()
                # gradient_reparam = [p.grad for p in layer_reparam.diffusion.model_net.parameters() if p.requires_grad]

                layer_resample = other_layers['layer_resample']
                z_star_resample = layer_resample(x, _y=y_preds.clone().detach())
                loss_resample = task_loss(z_star_resample, y, params)
                # loss.backward()
                # gradient_resample = [p.grad for p in layer.diffusion.model_net.parameters() if p.requires_grad]

                layer_distr_10 = other_layers['layer_distr_10']
                layer_distr_50 = other_layers['layer_distr_50']
                layer_distr_100 = other_layers['layer_distr_100']
                layer_distr_500 = other_layers['layer_distr_500']
                z_star_distr_10 = layer_distr_10(x, _y=y_preds.clone().detach())
                loss_distr_10 = task_loss(z_star_distr_10, y, params)
                z_star_distr_50 = layer_distr_50(x, _y=y_preds.clone().detach())
                loss_distr_50 = task_loss(z_star_distr_50, y, params)
                z_star_distr_100 = layer_distr_100(x, _y=y_preds.clone().detach())
                loss_distr_100 = task_loss(z_star_distr_100, y, params)
                z_star_distr_500 = layer_distr_500(x, _y=y_preds.clone().detach())
                loss_distr_500 = task_loss(z_star_distr_500, y, params)

                g_reparam = _named_grads(loss_reparam, layer_reparam.diffusion.model_net, retain_graph=True)
                g_resample = _named_grads(loss_resample, layer_resample.diffusion.model_net, retain_graph=True)
                
                g_distr_50 = _named_grads(loss_distr_50, layer_distr_50.diffusion.model_net, retain_graph=True)
                g_distr_100 = _named_grads(loss_distr_100, layer_distr_100.diffusion.model_net, retain_graph=True)
                g_distr_500 = _named_grads(loss_distr_500, layer_distr_500.diffusion.model_net, retain_graph=True)
                g_distr_10 = _named_grads(loss_distr_10, layer_distr_10.diffusion.model_net, retain_graph=True)

                overall_dp_10, overall_dp_10_norms = cos_by_param_from_grads(g_distr_10, g_reparam)
                overall_dp_50, overall_dp_50_norms = cos_by_param_from_grads(g_distr_50, g_reparam)
                overall_dp_100, overall_dp_100_norms = cos_by_param_from_grads(g_distr_100, g_reparam)
                overall_dp_500, overall_dp_500_norms = cos_by_param_from_grads(g_distr_500, g_reparam)
                overall_rp, overall_rp_norms = cos_by_param_from_grads(g_resample, g_reparam)

                print(f"cos_dp_10 = {overall_dp_10.item():.4f}, cos_dp_50 = {overall_dp_50.item():.4f}, cos_dp_100 = {overall_dp_100.item():.4f}, cos_dp_500 = {overall_dp_500.item():.4f}, cos_rp = {overall_rp.item():.4f}")
                print(f"cos_dp_10_norms = {overall_dp_10_norms.item():.4f}, cos_dp_50_norms = {overall_dp_50_norms.item():.4f}, cos_dp_100_norms = {overall_dp_100_norms.item():.4f}, cos_dp_500_norms = {overall_dp_500_norms.item():.4f}, cos_rp_norms = {overall_rp_norms.item():.4f}")

                # print(f"cos_dp_10 = {per_dp_10['net.0.weight'].item():.4f}, cos_dp_50 = {per_dp_50['net.0.weight'].item():.4f}, cos_dp_100 = {per_dp_100['net.0.weight'].item():.4f}, cos_dp_500 = {per_dp_500['net.0.weight'].item():.4f}, cos_rp = {per_rp['net.0.weight'].item():.4f}")
                grad_cnt += 1

                opt.zero_grad(set_to_none=True)
                z_star_reparam = layer_reparam(x, _y=y_preds)

                diffusion_loss = 0.0
                num_samples = 10
                for _ in range(num_samples):
                    t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=constants.DEVICE).long()
                    diffusion_loss += diffusion.diffusion_loss(y, t, x)
                diffusion_loss /= num_samples

                dfl_loss = task_loss_no_mean(z_star_reparam, y, params)

                if args.pretrain_epochs == 0:
                    dfl_grad_norm = get_grad_norm(dfl_loss.mean(), diffusion.model_net.parameters())
                    mse_grad_norm = get_grad_norm(diffusion_loss, diffusion.model_net.parameters())
                    # print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
                    alpha = 0.1
                    loss = alpha * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - alpha) * diffusion_loss
                    # loss = dfl_loss.mean()
                else:
                    dfl_grad_norm = get_grad_norm(dfl_loss.mean(), diffusion.model_net.parameters())
                    mse_grad_norm = get_grad_norm(diffusion_loss, diffusion.model_net.parameters())
                    # print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
                    alpha = 0.2
                    loss = alpha * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - alpha) * diffusion_loss
                    # loss = dfl_loss.mean()
                loss.backward()
            else:
                if idx is None:
                    z_star = layer(x)
                else:
                    z_star = layer(x, idx=idx, epoch=epoch)
                if "diffusion" in which:
                    diffusion_loss = 0.0
                    num_samples = 10
                    for _ in range(num_samples):
                        t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=constants.DEVICE).long()
                        diffusion_loss += diffusion.diffusion_loss(y, t, x)
                    diffusion_loss /= num_samples

                    dfl_loss = task_loss_no_mean(z_star, y, params)

                    # alpha = 0.9
                    # k = int((1 - alpha) * y.shape[0])
                    # tail_vals, _ = torch.topk(dfl_loss.sum(dim=1), k=k, largest=True)
                    # cvar = tail_vals.mean()
                    if args.pretrain_epochs == 0:
                        dfl_grad_norm = get_grad_norm(dfl_loss.mean(), diffusion.model_net.parameters())
                        mse_grad_norm = get_grad_norm(diffusion_loss, diffusion.model_net.parameters())
                        # print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
                        alpha = 0.1
                        loss = alpha * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - alpha) * diffusion_loss
                        # loss = dfl_loss.mean()
                    else:
                        dfl_grad_norm = get_grad_norm(dfl_loss.mean(), diffusion.model_net.parameters())
                        mse_grad_norm = get_grad_norm(diffusion_loss, diffusion.model_net.parameters())
                        # print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
                        alpha = 0.2
                        loss = alpha * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - alpha) * diffusion_loss
                        # loss = dfl_loss.mean()
                    loss.backward()
                else:
                    if idx is None:
                        z_star = layer(x)
                    else:
                        z_star = layer(x, idx=idx, epoch=epoch)
                    loss = _dfl_loss_no_mean(z_star, y, params)
                    loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(diffusion.model_net.parameters(), float('inf'))

            print(f"Epoch = {epoch}, train_df_loss = {dfl_loss.mean().item(): .4f}, grad_norm = {grad_norm.item(): .4f}")
            wandb.log({"train_df_loss": dfl_loss.mean().item(), "grad_norm": grad_norm.item()})
            opt.step()
            epoch_loss += dfl_loss.mean().item()
            cnt += 1

        time_end = time.time()
        total_training_time += time_end - time_start
        # print(f"Time for training = {time_end - time_start}")
        epoch_loss /= len(train_loader)
        wandb.log({"train_df_loss_epoch": epoch_loss})

        if args.pretrain_epochs > 0 or epoch % 10 == 0:
            diffusion.model_net.eval()
            layer.diffusion.model_net.eval()
            val_loss = 0.0
            num_samples = 0 
            with torch.no_grad():
                for batch in hold_loader:
                    if len(batch) == 2:
                        x, y = batch
                        idx = None
                    elif len(batch) == 3:
                        x, y, idx = batch

                    x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                    if idx is None:
                        z_star = layer(x)
                    else:
                        z_star = layer(x, idx=idx, epoch=-1)
                    loss = _dfl_loss_no_mean(z_star, y, params)
                    per_sample = loss.mean()
                    val_loss += per_sample.sum().item()
                    num_samples += y.shape[0]
                    print(f"Epoch = {epoch} | hold-out DFL = {per_sample.mean().item():.4f}")

            val_loss /= len(hold_loader)
            print(f"Epoch = {epoch} | final hold-out DFL = {val_loss:.4f}\n")
            wandb.log({"hold_out_df_loss": val_loss})

            test_loss = 0
            num_samples = 0 
            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 2:
                        x, y = batch
                        idx = None
                    elif len(batch) == 3:
                        x, y, idx = batch

                    x, y = x.to(constants.DEVICE), y.to(constants.DEVICE)
                    if idx is None:
                        z_star = layer(x)
                    else:
                        z_star = layer(x, idx=idx, epoch=-1)
                    loss = _dfl_loss_no_mean(z_star, y, params)
                    per_sample = loss.mean()
                    test_loss += per_sample.sum().item()
                    num_samples += y.shape[0]
                    print(f"Epoch = {epoch}, test_df_loss = {per_sample.mean().item(): .4f}")
                test_loss /= len(test_loader)
                print(f"\nEpoch = {epoch}, final test_df_loss = {test_loss: .4f}\n")
                wandb.log({"test_df_loss": test_loss})

            train_dfl_losses.append(epoch_loss)
            val_dfl_losses.append(val_loss)
            test_dfl_losses.append(test_loss)
            total_training_time_list.append(total_training_time)

        # -------- early-stopping -----
        min_delta = 1e-4
        if test_loss < best_val - min_delta:
            best_val     = test_loss
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
            if idx is None:
                z_star = layer(x)
            else:
                z_star = layer(x, idx=idx, epoch=-1)
            test_loss += _dfl_loss(z_star, y, params).mean().item()
        test_loss /= len(test_loader)
        print(f"\nFinal Epoch = {epoch}, test_df_loss = {test_loss: .4f}, final_training_time = {total_training_time}\n")
        wandb.log({"final_test_df_loss": test_loss})
    
    run_res = {
        'train_dfl_losses': train_dfl_losses,
        'val_dfl_losses': val_dfl_losses,
        'test_dfl_losses': test_dfl_losses,
        'total_training_time_list': total_training_time_list
    }
    torch.save(run_res, os.path.join(save_folder, '{}_run_res'.format(which)))
    torch.save(diffusion.model_net.state_dict(), os.path.join(save_folder, f'{which}.pth'))

    return diffusion, run_res

def eval_diffusion(which, diffusion, variables, params, mc_samples, save_folder, layer=None):
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
        
        layer = cvxpy_toy_parallel_kkt(params, mc_samples*y_pred_train.shape[0])
        z_star_train, _ = layer(y_pred_train.double().reshape(1, -1, params['n']))
        z_star_train = z_star_train[0]
        train_loss_task = task_loss(
            z_star_train, variables['Y_train_'], params)
        
        layer = cvxpy_toy_parallel_kkt(params, mc_samples*y_pred_test.shape[0])
        z_star_test, _ = layer(y_pred_test.double().reshape(1, -1, params['n']))
        z_star_test = z_star_test[0]
        test_loss_task = task_loss(
            z_star_test, variables['Y_test_'], params)

        layer = cvxpy_toy_parallel_kkt(params, mc_samples*y_pred_hold.shape[0])
        z_star_hold, _ = layer(y_pred_hold.double().reshape(1, -1, params['n']))
        z_star_hold = z_star_hold[0]
        hold_loss_task = task_loss(
            z_star_hold, variables['Y_hold_'], params)

        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))
        torch.save(hold_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_hold_task'.format(which)))

def eval_net(which, model, variables, params, save_folder, mc_samples=None, solver=None):
    if isinstance(model, model_classes.GaussianMLP):
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
        solver = cvxpy_toy_parallel_kkt(params, mc_samples*y_preds_train.shape[0])
        Y_sched_train, _ = solver(y_preds_train.double().reshape(1, -1, params['n']))
        Y_sched_train = Y_sched_train[0]
        train_loss_task = task_loss(
            Y_sched_train.float(), variables['Y_train_'], params)

        solver = cvxpy_toy_parallel_kkt(params, mc_samples*y_preds_test.shape[0])
        Y_sched_test, _ = solver(y_preds_test.double().reshape(1, -1, params['n']))
        Y_sched_test = Y_sched_test[0]
        test_loss_task = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)

        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))
    elif isinstance(model, model_classes.MLP):
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
        
        solver = cvxpy_toy_parallel_kkt(params, y_preds_train.shape[0])
        Y_sched_train, _ = solver(y_preds_train.double().reshape(1, -1, params['n']))
        Y_sched_train = Y_sched_train[0]
        train_loss_task = task_loss(
            Y_sched_train.float(), variables['Y_train_'], params)
        
        solver = cvxpy_toy_parallel_kkt(params, y_preds_test.shape[0])
        Y_sched_test, _ = solver(y_preds_test.double().reshape(1, -1, params['n']))
        Y_sched_test = Y_sched_test[0]
        test_loss_task = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)
        
        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))
    elif isinstance(model, model_classes.PolicyNet):
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
        
        train_loss_task = task_loss(
            y_preds_train.float(), variables['Y_train_'], params)
        
        test_loss_task = task_loss(
            y_preds_test.float(), variables['Y_test_'], params)
        
        torch.save(train_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_loss_task.detach().cpu().numpy(), 
            os.path.join(save_folder, '{}_test_task'.format(which)))
        
    print("test_loss_task = ", test_loss_task.item(), "test_rmse = ", test_rmse)

        
if __name__ == "__main__":
    from data_gen import generate_data, generate_data_multi_dim
    import random

    data_size = 1000
    logit_scale = 0.3
    batch_size = 32
    x_dim = 20
    y_dim = 10
    mc_samples = 30
    seed = 1

    DEVICE = "cuda:4"

    C = 2.0
    if y_dim == 1:
        C = 2.0
        X, Y, data_info = generate_data(n=data_size, x_dim=x_dim, seed=seed, C=C)
    else:
        C = 2.0
        X, Y, data_info = generate_data_multi_dim(n=data_size, x_dim=x_dim, y_dim=y_dim, seed=seed, C=C)

    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]
    if y_dim == 1:
        X_train_ = torch.tensor(X_train, dtype=torch.float, device=DEVICE)
        Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=DEVICE).unsqueeze(1)
        X_test_ = torch.tensor(X_test, dtype=torch.float, device=DEVICE)
        Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=DEVICE).unsqueeze(1)
    else:
        X_train_ = torch.tensor(X_train, dtype=torch.float, device=DEVICE)
        Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
        X_test_ = torch.tensor(X_test, dtype=torch.float, device=DEVICE)
        Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)
    variables_rmse = {'X_train_': X_train_, 'Y_train_': Y_train_, 
                'X_test_': X_test_, 'Y_test_': Y_test_}
    
    th_frac = 0.8
    inds = np.random.permutation(X_train.shape[0])
    train_inds = inds[ :int(X_train.shape[0] * th_frac)]
    hold_inds = inds[int(X_train.shape[0] * th_frac):]
    X_train2, X_hold2 = X_train[train_inds], X_train[hold_inds]
    Y_train2, Y_hold2 = Y_train[train_inds], Y_train[hold_inds]
    X_train2_ = torch.tensor(X_train2, dtype=torch.float32, device=DEVICE)
    if y_dim == 1:
        Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    else:
        Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE)
    X_hold2_ = torch.tensor(X_hold2, dtype=torch.float32, device=DEVICE)
    if y_dim == 1:
        Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    else:
        Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE)
    variables_task = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
            'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
            'X_test_': X_test_, 'Y_test_': Y_test_}
        
    params = {'n': y_dim, 'mc_samples': mc_samples, "data_info": data_info, 'logit_scale': logit_scale, 'C': C}
    YOUR_HOME_FOLDER = ""
    base_save = os.path.join(YOUR_HOME_FOLDER, "e2e-model-learning", "synthetic_example_new", f"synthetic_results_{data_size}_{y_dim}_pretrain")
    
    for run in range(0, 10):
        random.seed(seed + run)
        np.random.seed(seed + run)
        torch.manual_seed(seed + run)
        torch.cuda.manual_seed_all(seed + run)
        torch.backends.cudnn.deterministic = True

        save_folder = os.path.join(base_save, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1]).to(DEVICE)
        which = "task_net_mlp_gaussian_reparam"
        model_task.load_state_dict(torch.load(os.path.join(save_folder, f'{which}_{data_size}.pth')))
        layer = cvxpy_toy_parallel_kkt(params, mc_samples=mc_samples)
        eval_net(which, model_task, variables_task, params, save_folder, mc_samples=10)

        model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1]).to(DEVICE)
        which = "task_net_mlp_gaussian_distr"
        model_task.load_state_dict(torch.load(os.path.join(save_folder, f'{which}_{data_size}.pth')))
        layer = cvxpy_toy_parallel_kkt(params, mc_samples=mc_samples)
        eval_net(which, model_task, variables_task, params, save_folder, mc_samples=10)

        model_task = model_classes.MLP(X_train2_.shape[1], 1024, Y_train2_.shape[1]).to(DEVICE)
        which = "task_net_deterministic_mlp"
        model_task.load_state_dict(torch.load(os.path.join(save_folder, f'{which}_{data_size}.pth')))
        layer = cvxpy_toy_parallel_kkt(params, mc_samples=mc_samples)
        eval_net(which, model_task, variables_task, params, save_folder, mc_samples=10)

        diffusion_timesteps = 1000
        diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=DEVICE)
        # which = "diffusion_distr_est_resample"
        # diffusion.model_net.load_state_dict(torch.load(os.path.join(save_folder, f'diffusion_distr_est_resample_{data_size}.pth')))
        # eval_diffusion(which, diffusion, variables_task, params, mc_samples, save_folder)

        # which = "diffusion_distr_est"
        # diffusion.model_net.load_state_dict(torch.load(os.path.join(save_folder, f'diffusion_distr_est_{data_size}.pth')))
        # eval_diffusion(which, diffusion, variables_task, params, mc_samples, save_folder)

        # which = "diffusion_point_est"
        # diffusion.model_net.load_state_dict(torch.load(os.path.join(save_folder, f'diffusion_point_est_{data_size}.pth')))
        # eval_diffusion(which, diffusion, variables_task, params, mc_samples, save_folder)
