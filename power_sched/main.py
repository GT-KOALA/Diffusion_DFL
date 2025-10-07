#/usr/bin/env python3

import argparse
import setproctitle
import wandb

import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold 

from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

# try: import setGPU
# except ImportError: pass

import torch

import model_classes, nets, plot
from constants import *
from solver_mlp import MLPCvxpyModule, MLPCvxpyModule_distr

import sys
sys.path.append('.')
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

from diffusion_opt import Diffsion
from util import IndexedDataset, build_loader

def main():
    parser = argparse.ArgumentParser(
        description='Run electricity scheduling task net experiments.')
    parser.add_argument('--task', type=str, default='diffusion', help='rmse, weighted_rmse, task_net, task_net_ns, diffusion, diffusion_resample, two_stage_mlp, two_stage_gaussian, two_stage_diffusion, task_net_mlp_gaussian_qp, task_net_mlp_gaussian_reparam, task_net_mlp_gaussian_distr, task_net_deterministic_mlp')
    parser.add_argument('--save', type=str, metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--nRuns', type=int, default=10, metavar='runs', help='number of runs')
    parser.add_argument('--mc_samples', type=int, default=25, metavar='mc_samples', help='number of MC samples')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size', help='batch size')
    parser.add_argument('--cuda_device', type=int, default=7, metavar='cuda_device', help='cuda device')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='lr', help='learning rate for diffusion with score function')

    parser.add_argument('--use_distr_est', action='store_true', default=False, help='use distributional estimation')
    parser.add_argument('--sf_sample_size', type=int, default=10, metavar='sf_sample_size', help='sample size for score function')
    parser.add_argument('--pretrain_epochs', type=int, default=800, metavar='pretrain_epochs', help='number of pretraining epochs')
    parser.add_argument('--pretrain_model_path', type=str, default=None, metavar='pretrain_model_path', help='path to pretrained diffusion model')

    parser.add_argument('--seed', type=int, default=0, metavar='seed', help='random seed')
    parser.add_argument('--interval', type=int, default=1, metavar='interval', help='interval for replay buffer')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    args = parser.parse_args()

    time_str = dt.now().strftime("%Y%m%d_%H%M%S")
    
    use_wandb = getattr(args, "use_wandb", False)
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true" 
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init   = lambda *a, **k: None
        wandb.log    = lambda *a, **k: None
        wandb.finish = lambda *a, **k: None
        wandb.watch  = lambda *a, **k: None
        wandb.define_metric = lambda *a, **k: None
    else:
        wandb.login(key=YOUR_WANDB_API_KEY)

        if args.task == "diffusion" and args.use_distr_est:
            if not os.path.exists(f"wandb/ps_diffusion_distr_est"):
                os.makedirs(f"wandb/ps_diffusion_distr_est")
            wandb.init(project=f"ps_diffusion_distr_est", name=f"diffusion_distr_est_{time_str}", config=vars(args), dir=f"wandb/ps_diffusion_distr_est")
        elif args.task == "diffusion" and not args.use_distr_est:
            if not os.path.exists(f"wandb/ps_diffusion_point_est"):
                os.makedirs(f"wandb/ps_diffusion_point_est")
            wandb.init(project=f"ps_diffusion_point_est", name=f"diffusion_point_est_{time_str}", config=vars(args), dir=f"wandb/ps_diffusion_point_est")
        else:
            if not os.path.exists(f"wandb/ps_{args.task}"):
                os.makedirs(f"wandb/ps_{args.task}")
            wandb.init(project=f"ps_{args.task}", name=f"{args.task}_{time_str}", config=vars(args), dir=f"wandb/ps_{args.task}")

    if args.use_distr_est:
        if args.task == "diffusion_replay":
            from solver_distribution_replay import DiffusionCvxpyModuleReplay
        else:
            from solver_distribution_kkt import DiffusionCvxpyModule
    else:
        from solver_reparam import DiffusionCvxpyModule

    # if args.task != "diffusion":
    #     args.mc_samples = 1
    
    setproctitle.setproctitle('power_sched')

    if torch.cuda.is_available():
        DEVICE = f'cuda:{args.cuda_device}' if USE_GPU else 'cpu'
        set_device(args.cuda_device)
        print(f"Current device: {DEVICE}")

    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}
    # Load real data
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    X1, Y1 = load_data_with_features(os.path.join(dataset_dir, 'pjm_load_data_2008-11.txt'))
    X2, Y2 = load_data_with_features(os.path.join(dataset_dir, 'pjm_load_data_2012-16.txt'))
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    # Normalize features
    X[:,:-1] = (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)
    
    # Train, test split
    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]

    # Construct tensors (without intercepts).
    X_train_ = torch.tensor(X_train[:,:-1], dtype=torch.float, device=DEVICE)
    Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_test_ = torch.tensor(X_test[:,:-1], dtype=torch.float, device=DEVICE)
    Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)
    variables_rmse = {'X_train_': X_train_, 'Y_train_': Y_train_, 
                'X_test_': X_test_, 'Y_test_': Y_test_}

    if args.pretrain_epochs == 0:
        base_save = 'power_sched_results_no_pretrain' if args.save is None else '{}-results'.format(args.save)
    else:
        base_save = f'power_sched_results_final' if args.save is None else '{}-results'.format(args.save)
    for run in range(args.seed, args.nRuns + args.seed):
        print(f"Run {run}, seed {run}")
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        torch.backends.cudnn.deterministic = True

        save_folder = os.path.join(base_save, f"{run}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if args.task == 'rmse':
            # Run and eval rmse-minimizing net
            print("--------------------------------")
            print("Run rmse net.")
            print("--------------------------------")
            model_rmse = model_classes.Net(X_train[:,:-1], Y_train, [200, 200])
            if USE_GPU:
                model_rmse = model_rmse.to(DEVICE)
            model_rmse = nets.run_rmse_net(
                model_rmse, variables_rmse, X_train, Y_train)
            nets.eval_net("rmse_net", model_rmse, variables_rmse, params, save_folder)
        elif args.task == "weighted_rmse":
            # Run and eval task cost-weighted rmse-minimizing net (model defined/updated internally)
            print("--------------------------------")
            print("Run weighted rmse net.")
            print("--------------------------------")
            model_rmse_weighted = nets.run_weighted_rmse_net(X_train, Y_train, X_test, Y_test, params)
            nets.eval_net("weighted_rmse_net", model_rmse_weighted, variables_rmse, params, save_folder)
        else:
            th_frac = 0.8
            inds = np.random.permutation(X_train.shape[0])
            train_inds = inds[ :int(X_train.shape[0] * th_frac)]
            hold_inds = inds[int(X_train.shape[0] * th_frac):]
            X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
            Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]
            X_train2_ = torch.tensor(X_train2[:,:-1], dtype=torch.float32, device=DEVICE)
            Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE)
            X_hold2_ = torch.tensor(X_hold2[:,:-1], dtype=torch.float32, device=DEVICE)
            Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE)
            variables_task = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
                    'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
                    'X_test_': X_test_, 'Y_test_': Y_test_}
            if args.task == "task_net":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with MLP.")
                print("--------------------------------")
                model_task = model_classes.Net(X_train2[:,:-1], Y_train2, [200, 200])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                # model_task = nets.run_task_net(
                #     model_task, variables_task, params, X_train2, Y_train2, args)
                model_task = nets.run_task_net_dataloader(
                    model_task, train_loader, hold_loader, test_loader, params, which="task_net", save_folder=save_folder)
            
                nets.eval_net("task_net", model_task, variables_task, params, save_folder)
            elif args.task == "task_net_mlp_gaussian_qp":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with MLP.")
                print("--------------------------------")
                model_task = model_classes.Net(X_train2[:,:-1], Y_train2, [200, 200])
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=args.batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=args.batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=args.batch_size, shuffle=True)
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2)
                model_task = nets.run_weighted_rmse_net_mlp(
                    model_task, train_loader, hold_loader, test_loader, params, args)
            
                nets.eval_net("task_net_mlp_gaussian_qp", model_task, variables_task, params, save_folder, mc_samples=args.mc_samples)
            elif args.task == "task_net_mlp_gaussian_reparam":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Gaussian MLP.")
                print("--------------------------------")
                model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                # pretrain gaussian mlp
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2, set_sig=False, training_steps=1000)
                layer = MLPCvxpyModule(model_task, params, args.mc_samples, distr_est=args.use_distr_est)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                model_task = nets.run_task_net_mlp(
                    model_task, train_loader, hold_loader, test_loader, layer, params, args, "task_net_mlp_gaussian_reparam", save_folder)
            
                nets.eval_net("task_net_mlp_gaussian_reparam", model_task, variables_task, params, save_folder, mc_samples=args.mc_samples)
            elif args.task == "task_net_mlp_gaussian_distr":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Gaussian MLP.")
                print("--------------------------------")
                model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                # pretrain gaussian mlp
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2, set_sig=False, training_steps=1000)
                layer = MLPCvxpyModule_distr(model_task, params, args.mc_samples, distr_est=args.use_distr_est)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                model_task = nets.run_task_net_mlp(
                    model_task, train_loader, hold_loader, test_loader, layer, params, args, "task_net_mlp_gaussian_distr", save_folder)
            
                nets.eval_net("task_net_mlp_gaussian_distr", model_task, variables_task, params, save_folder, mc_samples=args.mc_samples)
            elif args.task == "task_net_deterministic_mlp":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with MLP.")
                print("--------------------------------")
                model_task = model_classes.MLP(X_train2_.shape[1], 1024, Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2, set_sig=False)
                args.mc_samples = 1 # no randomness for MLP
                layer = MLPCvxpyModule(model_task, params, args.mc_samples, distr_est=args.use_distr_est)

                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                model_task = nets.run_task_net_mlp(
                    model_task, train_loader, hold_loader, test_loader, layer, params, args, "task_net_deterministic_mlp", save_folder)
            
                nets.eval_net("task_net_deterministic_mlp", model_task, variables_task, params, save_folder)
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

                diffusion.pretrain_diffusion(X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, args, two_stage=True)
            
                nets.eval_diffusion("two_stage_diffusion", diffusion, variables_task, params, args.mc_samples, save_folder)
            elif args.task == "two_stage_gaussian":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Two-Stage Gaussian.")
                print("--------------------------------")
                model_task = model_classes.GaussianMLP(X_train2_.shape[1], [1024, 1024], Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2, training_steps=args.pretrain_epochs)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                # model_task = nets.run_two_stage(
                #     model_task, train_loader, hold_loader, test_loader, variables_task, params)
            
                nets.eval_net("two_stage_gaussian", model_task, variables_task, params, save_folder, mc_samples=args.mc_samples)
            elif args.task == "two_stage_mlp":
                # Run and eval task-minimizing net, building off rmse net results.
                print("--------------------------------")
                print("Run task net with Two-Stage MLP.")
                print("--------------------------------")
                model_task = model_classes.MLP(X_train2_.shape[1], 1024, Y_train2_.shape[1])
                if USE_GPU:
                    model_task = model_task.to(DEVICE)
                model_task = nets.run_rmse_net(
                    model_task, variables_task, X_train2, Y_train2, set_sig=False, training_steps=args.pretrain_epochs)
                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)
                # model_task = nets.run_two_stage(
                #     model_task, train_loader, hold_loader, test_loader, variables_task, params)
            
                nets.eval_net("two_stage_mlp", model_task, variables_task, params, save_folder, mc_samples=1)
            elif args.task == "diffusion" or args.task == "diffusion_resample":
                print("--------------------------------")
                print("Run task net with Diffusion.")
                print("--------------------------------")
                diffusion_timesteps = 1000
                # diffusion_timesteps = 300
                diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, beta_schedule="linear", device=DEVICE, sf_sample_size=args.sf_sample_size)

                batch_size = args.batch_size
                # fold_loaders = []
                # full_ds = torch.utils.data.TensorDataset(X_train_, Y_train_)
                # for fold, (train_inds, hold_inds) in enumerate(KFold(n_splits=10, shuffle=True, random_state=run).split(X_train)):
                #     train_loader = build_loader(full_ds, train_inds, shuffle_flag=True, batch_size=batch_size)
                #     hold_loader = build_loader(full_ds, hold_inds, shuffle_flag=True, batch_size=batch_size)
                #     fold_loaders.append((train_loader, hold_loader))
                # test_loader = build_loader(torch.utils.data.TensorDataset(X_test_, Y_test_), range(len(X_test_)), shuffle_flag=True, batch_size=batch_size)
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                # pretrain diffusion
                diffusion.pretrain_diffusion(X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, args)

                layer = DiffusionCvxpyModule(diffusion, params, args.mc_samples, distr_est=args.use_distr_est, resample=args.task == "diffusion_resample")

                if args.use_distr_est and args.task == "diffusion":
                    which = "diffusion_distr_est"
                elif args.task == "diffusion_resample":
                    which = "diffusion_distr_est_resample"
                else:
                    which = "diffusion_point_est"

                # nets.run_task_net_diffusion(diffusion, fold_loaders, test_loader, layer, params, args, which, save_folder)
                nets.run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, layer, params, args, which, save_folder)
                torch.save(diffusion.model_net.state_dict(), os.path.join(save_folder, f'{which}.pth'))
                nets.eval_diffusion(which, diffusion, variables_task, params, args.mc_samples, save_folder)
            elif args.task == "diffusion_replay":
                print("--------------------------------")
                print("Run task net with Diffusion with replay.")
                print("--------------------------------")
                diffusion_timesteps = 1000
                diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=DEVICE, sf_sample_size=args.sf_sample_size)

                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(IndexedDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(IndexedDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(IndexedDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                diffusion.pretrain_diffusion(X_train2_, Y_train2_, hold_loader, base_save, args)

                layer = DiffusionCvxpyModuleReplay(diffusion, params, args.mc_samples, interval=args.interval, distr_est=args.use_distr_est)
                # nets.run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, entire_train_loader, entire_hold_loader, entire_test_loader, layer, params, args, which="diffusion_replay", save_folder=save_folder)
                nets.run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, layer, params, args, which="diffusion_replay", save_folder=save_folder)
                torch.save(diffusion.model_net.state_dict(), os.path.join(save_folder, 'diffusion_replay.pth'))
                # nets.eval_diffusion(diffusion, variables_task, layer, params, save_folder)
            elif args.task == "diffusion_with_time":
                print("--------------------------------")
                print("Run task net with Diffusion with time.")
                print("--------------------------------")                
                nets.run_task_net_diffusion_with_time(variables_task, params, args)
            elif args.task == "diffusion_grad_comparison":
                print("--------------------------------")
                print("Run task net with Diffusion with gradient comparison.")
                print("--------------------------------")
                diffusion_timesteps = 1000
                diffusion = Diffsion(x_dim=X_train2_.shape[1], y_dim=Y_train2_.shape[1], timesteps=diffusion_timesteps, device=DEVICE, sf_sample_size=args.sf_sample_size)

                batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train2_, Y_train2_), batch_size=batch_size, shuffle=True)
                hold_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_hold2_, Y_hold2_), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_, Y_test_), batch_size=batch_size, shuffle=True)

                # pretrain diffusion
                diffusion.pretrain_diffusion(X_train2_, Y_train2_, X_hold2_, Y_hold2_, base_save, args)

                from solver_distribution_kkt import DiffusionCvxpyModule as DiffusionCvxpyModule_distr
                from solver_reparam_kkt import DiffusionCvxpyModule as DiffusionCvxpyModule_reparam

                layer_distr_10 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=10, distr_est=True, resample=False)
                layer_distr_50 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=50, distr_est=True, resample=False)
                layer_distr_100 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=100, distr_est=True, resample=False)
                layer_distr_500 = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, num_iter=500, distr_est=True, resample=False)
                layer_resample = DiffusionCvxpyModule_distr(diffusion, params, args.mc_samples, distr_est=True, resample=True)
                layer_reparam = DiffusionCvxpyModule_reparam(diffusion, params, args.mc_samples, distr_est=False, resample=False)
                which = f"diffusion_grad_comparison"
                nets.run_task_net_diffusion(diffusion, train_loader, hold_loader, test_loader, layer_distr_10, params, args, which=which, other_layers={'layer_distr_10': layer_distr_10, 'layer_distr_50': layer_distr_50, 'layer_distr_100': layer_distr_100, 'layer_distr_500': layer_distr_500, 'layer_resample': layer_resample, 'layer_reparam': layer_reparam}, save_folder=save_folder)
        
    # plot.plot_results(map(
    #     lambda x: os.path.join(base_save, str(x)), range(args.nRuns)), base_save)

def f_obj(z, y, params):
    gamma_under = params["gamma_under"]
    gamma_over = params["gamma_over"]
    return (gamma_under * torch.clamp(y - z, min=0) + 
            gamma_over * torch.clamp(z - y, min=0) + 
            0.5 * (z - y)**2).mean(0)

def comp_true_obj(Y_test, layer, params):    
    z_star_tuple, _ = layer(Y_test.unsqueeze(1))
    z_stars_regret = z_star_tuple[0]
    f_evals_regret = f_obj(z_stars_regret, Y_test, params)

    return f_evals_regret, z_stars_regret

def load_data_with_features(filename):
    tz = pytz.timezone("America/New_York")
    df = pd.read_csv(filename, sep=" ", header=None, usecols=[1,2,3], 
        names=["time","load","temp"])
    df["time"] = df["time"].apply(dt.fromtimestamp, tz=tz)
    df["date"] = df["time"].apply(lambda x: x.date())
    df["hour"] = df["time"].apply(lambda x: x.hour)
    df.drop_duplicates("time", inplace=True)

    # Create one-day tables and interpolate missing entries
    df_load = df.pivot(index="date", columns="hour", values="load")
    df_temp = df.pivot(index="date", columns="hour", values="temp")
    df_load = df_load.transpose().fillna(method="backfill").transpose()
    df_load = df_load.transpose().fillna(method="ffill").transpose()
    df_temp = df_temp.transpose().fillna(method="backfill").transpose()
    df_temp = df_temp.transpose().fillna(method="ffill").transpose()

    holidays = USFederalHolidayCalendar().holidays(
        start='2008-01-01', end='2016-12-31').to_pydatetime()
    holiday_dates = set([h.date() for h in holidays])

    s = df_load.reset_index()["date"]
    data={"weekend": s.apply(lambda x: x.isoweekday() >= 6).values,
          "holiday": s.apply(lambda x: x in holiday_dates).values,
          "dst": s.apply(lambda x: tz.localize(
            dt.combine(x, dt.min.time())).dst().seconds > 0).values,
          "cos_doy": s.apply(lambda x: np.cos(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "sin_doy": s.apply(lambda x: np.sin(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values}
    df_feat = pd.DataFrame(data=data, index=df_load.index)

    # Construct features and normalize (all but intercept)
    X = np.hstack([df_load.iloc[:-1].values,        # past load
                    df_temp.iloc[:-1].values,       # past temp
                    df_temp.iloc[:-1].values**2,    # past temp^2
                    df_temp.iloc[1:].values,        # future temp
                    df_temp.iloc[1:].values**2,     # future temp^2
                    df_temp.iloc[1:].values**3,     # future temp^3
                    df_feat.iloc[1:].values,        
                    np.ones((len(df_feat)-1, 1))]).astype(np.float64)
    # X[:,:-1] = \
    #     (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)

    Y = df_load.iloc[1:].values

    return X, Y


if __name__=='__main__':
    main()