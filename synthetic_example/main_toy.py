#/usr/bin/env python3

import argparse
import setproctitle
import wandb
import copy
import os
import random
import pandas as pd
import numpy as np

from datetime import datetime as dt

import torch

import model_classes, nets_toy
from constants import *
from solver_mlp import MLPCvxpyModule, MLPCvxpyModule_distr
from data_gen import gen_toy_data, comp_true_obj, gen_toy_data_simple, generate_data, z_star_empirical, generate_data_multi_dim
from diffusion_opt import Diffsion

from util import IndexedDataset

import sys
sys.path.append('.')
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description='Run electricity scheduling task net experiments.')
    parser.add_argument('--task', type=str, default='diffusion', help='rmse, weighted_rmse, task_net, task_net_ns, diffusion, diffusion_resample, two_stage_mlp, two_stage_gaussian, two_stage_diffusion, task_net_mlp_gaussian_reparam, task_net_mlp_gaussian_distr, task_net_deterministic_mlp')
    parser.add_argument('--save', type=str, metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--nRuns', type=int, default=10, metavar='runs', help='number of runs')
    parser.add_argument('--mc_samples', type=int, default=30, metavar='mc_samples', help='number of MC samples')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='batch size')
    parser.add_argument('--cuda_device', type=int, default=7, metavar='cuda_device', help='cuda device')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='lr', help='learning rate for diffusion with score function')

    parser.add_argument('--use_distr_est', action='store_true', default=False, help='use distributional estimation')

    parser.add_argument('--x_dim', type=int, default=20, metavar='x_dim', help='x dimension')
    parser.add_argument('--y_dim', type=int, default=10, metavar='y_dim', help='y dimension')
    parser.add_argument('--logit_scale', type=float, default=0.3, metavar='logit_scale', help='logit scale')
    parser.add_argument('--data_size', type=int, default=1000, metavar='data_size', help='number of data points')
    parser.add_argument('--pretrain_epochs', type=int, default=1000, metavar='pretrain_epochs', help='number of pretraining epochs')

    parser.add_argument('--pretrain_model_path', type=str, default=None, metavar='pretrain_model_path', help='path to pretrained diffusion model')
    parser.add_argument('--interval', type=int, default=1, metavar='interval', help='interval for replay buffer')

    parser.add_argument('--seed', type=int, default=1, metavar='seed', help='random seed')
    args = parser.parse_args()

    time_str = dt.now().strftime("%Y%m%d_%H%M%S")
    wandb.login(key=YOUR_WANDB_API_KEY)

    if args.task == "diffusion" and args.use_distr_est:
        if not os.path.exists(f"wandb/diffusion_distr_est"):
            os.makedirs(f"wandb/diffusion_distr_est")
        wandb.init(project=f"synthetic_diffusion_distr_est", name=f"diffusion_distr_est_{time_str}", config=vars(args), dir=f"wandb/diffusion_distr_est")
    elif args.task == "diffusion" and not args.use_distr_est:
        if not os.path.exists(f"wandb/diffusion_point_est"):
            os.makedirs(f"wandb/diffusion_point_est")
        wandb.init(project=f"synthetic_diffusion_point_est", name=f"diffusion_point_est_{time_str}", config=vars(args), dir=f"wandb/diffusion_point_est")
    else:
        if not os.path.exists(f"wandb/{args.task}"):
            os.makedirs(f"wandb/{args.task}")
        wandb.init(project=f"synthetic_diffusion_{args.task}", name=f"{args.task}_{time_str}", config=vars(args), dir=f"wandb/{args.task}")
    
    if args.use_distr_est:
        from solver_distribution_kkt import DiffusionCvxpyModule
    else:
        from solver_reparam_kkt import DiffusionCvxpyModule

    # if args.task != "diffusion":
    #     args.mc_samples = 1
    
    setproctitle.setproctitle('synthetic')

    if torch.cuda.is_available():
        DEVICE = f'cuda:{args.cuda_device}' if USE_GPU else 'cpu'
        set_device(args.cuda_device)
        print(f"Current device: {DEVICE}")

    # Generate dummy data
    print("Using dummy data")
    # logit_scale = args.logit_scale
    # X, Y, data_info = gen_toy_data(
    #     m=args.data_size,
    #     x_dim=args.x_dim,
    #     a=a, b=b, p=p0,
    #     noise_std=noise_std,
    #     seed=args.seed,
    #     x_dependent_prob=x_dep,
    #     logit_scale=logit_scale,
    # )

    is_simple = False

    if is_simple:
        X_train2_, Y_train_2, data_info = gen_toy_data_simple(m=args.batch_size * 2, x_dim=args.x_dim, x_dependent_prob=True, seed=args.seed)
        X_hold2_, Y_hold2_, _ = gen_toy_data_simple(m=args.batch_size * 10, x_dim=args.x_dim, x_dependent_prob=True, seed=args.seed + 1)
        X_test, Y_test, _ = gen_toy_data_simple(m=args.batch_size * 10, x_dim=args.x_dim, x_dependent_prob=True, seed=args.seed + 2)
        # X_hold2_ = copy.deepcopy(X_train2_).repeat(5, 0)
        # X_test = copy.deepcopy(X_train2_).repeat(5, 0)
        X_train2_ = torch.tensor(X_train2_[:,:], dtype=torch.float32, device=DEVICE)
        Y_train2_ = torch.tensor(Y_train_2, dtype=torch.float32, device=DEVICE)
        X_hold2_ = torch.tensor(X_hold2_[:,:], dtype=torch.float32, device=DEVICE)
        Y_hold2_ = torch.tensor(Y_hold2_, dtype=torch.float32, device=DEVICE)
        X_test_ = torch.tensor(X_test[:,:], dtype=torch.float32, device=DEVICE)
        Y_test_ = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE)
        variables_task = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
                    'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
                    'X_test_': X_test_, 'Y_test_': Y_test_}
    else:
        if args.y_dim == 1:
            C = 2.0
            X, Y, data_info = generate_data(n=args.data_size, x_dim=args.x_dim, seed=args.seed, C=C)
        else:
            C = 2.0
            X, Y, data_info = generate_data_multi_dim(n=args.data_size, x_dim=args.x_dim, y_dim=args.y_dim, seed=args.seed, C=C)

        n_tt = int(len(X) * 0.8)
        X_train, Y_train = X[:n_tt], Y[:n_tt]
        X_test, Y_test = X[n_tt:], Y[n_tt:]
        if args.y_dim == 1:
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
        if args.y_dim == 1:
            Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        else:
            Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE)
        X_hold2_ = torch.tensor(X_hold2, dtype=torch.float32, device=DEVICE)
        if args.y_dim == 1:
            Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        else:
            Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE)
        variables_task = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
                'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
                'X_test_': X_test_, 'Y_test_': Y_test_}
        
    params = {'n': args.y_dim, 'mc_samples': args.mc_samples, "data_info": data_info, 'C': C, 'x_dim': args.x_dim, 'batch_size': args.batch_size}
    
    if args.pretrain_epochs == 0:
        base_save = f'synthetic_{args.data_size}_{args.y_dim}_no_pretrain' if args.save is None else '{}-results'.format(args.save)
    else:
        base_save = f'synthetic_results_{args.data_size}_{args.y_dim}_pretrain' if args.save is None else '{}-results'.format(args.save)

    for run in range(args.nRuns):
        random.seed(args.seed + run)
        np.random.seed(args.seed + run)
        torch.manual_seed(args.seed + run)
        torch.cuda.manual_seed_all(args.seed + run)
        torch.backends.cudnn.deterministic = True

        save_folder = os.path.join(base_save, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if args.task == 'rmse':
            # Run and eval rmse-minimizing net
            print("--------------------------------")
            print("Run rmse net.")
            print("--------------------------------")
            model_rmse = model_classes.Net(X_train, Y_train, [1024, 1024])
            if USE_GPU:
                model_rmse = model_rmse.to(DEVICE)
            model_rmse = nets_toy.run_rmse_net(
                model_rmse, variables_rmse, X_train, Y_train)
            nets_toy.eval_net("rmse_net", model_rmse, variables_rmse, params, save_folder)
        elif args.task == "weighted_rmse":
            # Run and eval task cost-weighted rmse-minimizing net (model defined/updated internally)
            print("--------------------------------")
            print("Run weighted rmse net.")
            print("--------------------------------")
            model_rmse_weighted = nets_toy.run_weighted_rmse_net(X_train, Y_train, X_test, Y_test, params)
            nets_toy.eval_net("weighted_rmse_net", model_rmse_weighted, variables_rmse, params, save_folder)
        else:
            # f_evals_regret, f_evals_env, z_star_evals_regret, z_star_evals_env = comp_true_obj(X_test_, Y_test_, params, seed=args.seed, device=DEVICE, simple=is_simple)
            # print(f"True obj: {f_evals_regret.mean():.4f}, {f_evals_env.mean():.4f}")
            # wandb.log({"true_obj_regret": f_evals_regret.mean(), "true_obj_env": f_evals_env.mean()})

            if args.task == "task_net_mlp_gaussian_qp":
                print("--------------------------------")
                print("Run task net with MLP.")
                print("--------------------------------")
                model_task = model_classes.Net(X_train2, Y_train2, [1024, 1024])
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=args.batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=args.batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=args.batch_size, shuffle=True)
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets_toy.run_weighted_rmse_net_mlp(
                    model_task, train_loader, hold_loader, test_loader, params, args)
            
                nets_toy.eval_net("task_net_mlp_gaussian_qp", model_task, variables_task, params, save_folder)
            elif args.task == "task_net_mlp_gaussian_reparam":
                print("--------------------------------")
                print("Run task net with Gaussian MLP.")
                print("--------------------------------")
                model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)

                # pretrain gaussian mlp
                if args.pretrain_epochs > 0:
                    model_task = nets_toy.run_rmse_net(
                        model_task, variables_task, set_sig=False, pretrain_epochs=args.pretrain_epochs)
                    torch.save(model_task.state_dict(), os.path.join(save_folder, f'pretrain_model_task_net_mlp_gaussian_reparam_{args.pretrain_epochs}.pth'))
                layer = MLPCvxpyModule(model_task, params, args.mc_samples)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                mlp_model, run_res = nets_toy.run_task_net_gaussian(
                    model_task, train_loader, hold_loader, test_loader, layer, params, args, which=f"task_net_mlp_gaussian_reparam_{args.data_size}", save_folder=save_folder)
                torch.save(mlp_model.state_dict(), os.path.join(save_folder, f'task_net_mlp_gaussian_reparam_{args.data_size}.pth'))
                nets_toy.eval_net("task_net_mlp_gaussian_reparam", model_task, variables_task, params, mc_samples=args.mc_samples, save_folder=save_folder)
            elif args.task == "task_net_mlp_gaussian_distr":
                print("--------------------------------")
                print("Run task net with Gaussian MLP.")
                print("--------------------------------")
                model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                # pretrain gaussian mlp
                if args.pretrain_epochs > 0:
                    model_task = nets_toy.run_rmse_net(
                        model_task, variables_task, set_sig=False, pretrain_epochs=args.pretrain_epochs)
                    torch.save(model_task.state_dict(), os.path.join(save_folder, f'pretrain_model_task_net_mlp_gaussian_distr_{args.pretrain_epochs}.pth'))
                layer = MLPCvxpyModule_distr(model_task, params, args.mc_samples)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                mlp_model, run_res = nets_toy.run_task_net_gaussian(
                    model_task, train_loader, hold_loader, test_loader, layer, params, args, which=f"task_net_mlp_gaussian_distr_{args.data_size}", save_folder=save_folder)
                nets_toy.eval_net("task_net_mlp_gaussian_distr", model_task, variables_task, params, mc_samples=args.mc_samples, save_folder=save_folder)
                torch.save(mlp_model.state_dict(), os.path.join(save_folder, f'task_net_mlp_gaussian_distr_{args.data_size}.pth'))
            elif args.task == "task_net_deterministic_mlp":
                print("--------------------------------")
                print("Run task net with deterministic MLP.")
                print("--------------------------------")
                model_task = model_classes.MLP(X_train2_.shape[1], 1024, Y_train2_.shape[1]).to(DEVICE)
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                if args.pretrain_epochs > 0:
                    args.pretrain_epochs = args.pretrain_epochs
                    model_task = nets_toy.run_rmse_net(
                        model_task, variables_task, set_sig=False, pretrain_epochs=args.pretrain_epochs)
                    torch.save(model_task.state_dict(), os.path.join(save_folder, f'pretrain_model_task_net_deterministic_mlp_{args.pretrain_epochs}.pth'))
                args.mc_samples = 1 # no sampling for MLP, deterministic optimization
                layer = MLPCvxpyModule(model_task, params, args.mc_samples)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                mlp_model, run_res = nets_toy.run_task_net_gaussian(
                    model_task, train_loader, hold_loader, test_loader, layer, params, args, which=f"task_net_deterministic_mlp_{args.data_size}", save_folder=save_folder)
            
                nets_toy.eval_net("task_net_deterministic_mlp", model_task, variables_task, params, mc_samples=args.mc_samples, save_folder=save_folder)
                torch.save(mlp_model.state_dict(), os.path.join(save_folder, f'task_net_deterministic_mlp_{args.data_size}.pth'))
            elif args.task == "two_stage_diffusion":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Two-Stage Diffusion.")
                print("--------------------------------")
                diffusion_timesteps = 1000
                diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=DEVICE)
                
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                # pretrain diffusion
                info = {"data_size": args.data_size, "logit_scale": args.logit_scale, "pretrain_epochs": args.pretrain_epochs, "x_dim": args.x_dim, "y_dim": args.y_dim}
                diffusion.pretrain_diffusion(X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, info)

                # model_task = nets_toy_cubic.run_two_stage(
                #     diffusion, train_loader, hold_loader, test_loader, variables_task, params)
            
                which = "two_stage_diffusion"
                nets_toy.eval_diffusion(which, diffusion, variables_task, params, args.mc_samples, save_folder=save_folder)
            elif args.task == "two_stage_gaussian":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Two-Stage Gaussian.")
                print("--------------------------------")
                model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1]).to(DEVICE)
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets_toy.run_rmse_net(
                    model_task, variables_task, set_sig=False, pretrain_epochs=args.pretrain_epochs)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
            
                nets_toy.eval_net("two_stage_gaussian", model_task, variables_task, params, save_folder, mc_samples=args.mc_samples)
            elif args.task == "two_stage_mlp":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Two-Stage MLP.")
                print("--------------------------------")
                model_task = model_classes.MLP(X_train2_.shape[1], 1024, Y_train2_.shape[1]).to(DEVICE)
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets_toy.run_rmse_net(
                    model_task, variables_task, set_sig=False, pretrain_epochs=args.pretrain_epochs)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
            
                nets_toy.eval_net("two_stage_mlp", model_task, variables_task, params, save_folder)
            elif args.task == "diffusion" or args.task == "diffusion_resample":
                print("--------------------------------")
                print("Run task net with Diffusion.")
                print("--------------------------------")
                diffusion_timesteps = 1000
                diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=DEVICE)

                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                # pretrain diffusion
                info = {"data_size": args.data_size, "logit_scale": args.logit_scale, "pretrain_epochs": args.pretrain_epochs, "x_dim": args.x_dim, "y_dim": args.y_dim}
                diffusion.pretrain_diffusion(X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, info)

                layer = DiffusionCvxpyModule(diffusion, params, args.mc_samples, distr_est=args.use_distr_est, resample=args.task == "diffusion_resample")
                if args.use_distr_est:
                    if args.task == "diffusion_resample":
                        which = f"diffusion_distr_est_resample_{args.data_size}"
                    else:
                        which = f"diffusion_distr_est_{args.data_size}"
                else:
                    if args.task == "diffusion_resample":
                        which = f"diffusion_point_est_resample_{args.data_size}"
                    else:
                        which = f"diffusion_point_est_{args.data_size}"
                nets_toy.run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, layer, params, args, which=which, save_folder=save_folder)
                torch.save(diffusion.model_net.state_dict(), os.path.join(save_folder, f'{which}.pth'))
                nets_toy.eval_diffusion(which, diffusion, variables_task, params, args.mc_samples, save_folder)
            elif args.task == "diffusion_grad_comparison":
                print("--------------------------------")
                print("Run task net with Diffusion with gradient comparison.")
                print("--------------------------------")
                diffusion_timesteps = 1000
                diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=DEVICE)

                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                # pretrain diffusion
                info = {"data_size": args.data_size, "logit_scale": args.logit_scale, "pretrain_epochs": args.pretrain_epochs, "x_dim": args.x_dim, "y_dim": args.y_dim}
                diffusion.pretrain_diffusion(X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, info)

                from solver_distribution_kkt import DiffusionCvxpyModule as DiffusionCvxpyModule_distr
                from solver_reparam_kkt import DiffusionCvxpyModule as DiffusionCvxpyModule_reparam

                layer_distr_10 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=10, distr_est=True, resample=False)
                layer_distr_50 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=50, distr_est=True, resample=False)
                layer_distr_100 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=100, distr_est=True, resample=False)
                layer_distr_500 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=500, distr_est=True, resample=False)
                layer_resample = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, distr_est=True, resample=True)
                layer_reparam = DiffusionCvxpyModule_reparam(diffusion, params, args.mc_samples, distr_est=False, resample=False)
                which = f"diffusion_grad_comparison_{args.data_size}"
                nets_toy.run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, layer_distr_10, params, args, which=which, other_layers={'layer_distr_10': layer_distr_10, 'layer_distr_50': layer_distr_50, 'layer_distr_100': layer_distr_100, 'layer_distr_500': layer_distr_500, 'layer_resample': layer_resample, 'layer_reparam': layer_reparam}, save_folder=save_folder)
    # plot.plot_results(map(
    #     lambda x: os.path.join(base_save, str(x)), range(args.nRuns)), base_save)


if __name__=='__main__':
    main()