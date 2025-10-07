import os
import sys
import pandas as pd
import torch
import numpy as np
import wandb
import copy
import cvxpy as cp
import random
import argparse
import tqdm
import time
import datetime as dt
import datetime
import setproctitle

from cvxpylayers.torch import CvxpyLayer

from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import SP500DataLoader, generateDataset
from model import PortfolioModel, CovarianceModel, MLP, GaussianMLP
from diffusion_opt import Diffsion
from solver_distribution_kkt import DiffusionCvxpyModule
from solver_mlp import MLPCvxpyModule, MLPCvxpyModule_distr
from portfolio_utils_diffusion import train_portfolio_diffusion, validate_portfolio_diffusion, test_portfolio_diffusion, eval_diffusion
from portfolio_utils_mlp import train_portfolio_mlp, validate_portfolio_mlp, test_portfolio_mlp, run_rmse_net, eval_net
from portfolio_utils import train_portfolio, validate_portfolio, test_portfolio
from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection 

if __name__ == '__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('--filepath', type=str, default='', help='filename under folder results')
    parser.add_argument('--task', type=str, default='diffusion', help='rmse, weighted_rmse, task_net, task_net_ns, diffusion, diffusion_resample, two_stage_mlp, two_stage_gaussian, two_stage_diffusion, task_net_mlp_gaussian_qp, task_net_gaussian_reparam, task_net_gaussian_distr, task_net_deterministic_mlp')
    parser.add_argument('--method', type=str, default='decision-focused', help='rmse, weighted_rmse, task_net, task_net_ns, diffusion, two_stage_mlp, two_stage_diffusion, task_net_mlp_gaussian_qp, task_net_deterministic_mlp')
    parser.add_argument('--nRuns', type=int, default=5, metavar='runs', help='number of runs')

    parser.add_argument('--T-size', type=int, default=10, help='the size of reparameterization metrix')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--n', type=int, default=50, help='number of securities')
    parser.add_argument('--num-samples', type=int, default=0, help='number of samples, 0 -> all')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size', help='batch size')

    parser.add_argument('--cuda_device', type=int, default=7, help='cuda device')
    parser.add_argument('--mc_samples', type=int, default=10, help='number of samples for distribution estimation')
    parser.add_argument('--pretrain_epochs', type=int, default=300, help='number of pretraining epochs')

    parser.add_argument('--use_distr_est', action='store_true', default=False, help='use distributional estimation')

    parser.add_argument('--seed', type=int, default=2, metavar='seed', help='random seed')
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
        if args.use_distr_est and args.task == "diffusion":
            if not os.path.exists(f"wandb/stock_portfolio_diffusion_distr_est"):
                os.makedirs(f"wandb/stock_portfolio_diffusion_distr_est")
            wandb.init(project=f"stock_portfolio_diffusion_distr_est", name=f"diffusion_distr_est_{time_str}", config=vars(args), dir=f"wandb/stock_portfolio_diffusion_distr_est")
        elif not args.use_distr_est and args.task == "diffusion":
            if not os.path.exists(f"wandb/stock_portfolio_diffusion_point_est"):
                os.makedirs(f"wandb/stock_portfolio_diffusion_point_est")
            wandb.init(project=f"stock_portfolio_diffusion_point_est", name=f"diffusion_point_est_{time_str}", config=vars(args), dir=f"wandb/stock_portfolio_diffusion_point_est")
        else:
            if not os.path.exists(f"wandb/stock_portfolio_{args.task}"):
                os.makedirs(f"wandb/stock_portfolio_{args.task}")
            wandb.init(project=f"stock_portfolio_{args.task}", name=f"{args.task}_{time_str}", config=vars(args), dir=f"wandb/stock_portfolio_{args.task}")

    portfolio_opt_dir = os.path.abspath(os.path.dirname(__file__))
    print("portfolio_opt_dir:", portfolio_opt_dir)

    sp500_data_dir = os.path.join(portfolio_opt_dir, "data", "sp500")
    sp500_data = SP500DataLoader(sp500_data_dir, "sp500",
                                start_date=dt.datetime(2004, 1, 1),
                                end_date=dt.datetime(2017, 1, 1),
                                collapse="daily",
                                overwrite=False,
                                verbose=True)

    if args.use_distr_est:
        from solver_distribution_kkt import DiffusionCvxpyModule
    else:
        from solver_reparam import DiffusionCvxpyModule
    
    setproctitle.setproctitle('stock_portfolio')

    filepath = args.filepath
    seed = 0
    n = args.n
    num_samples = args.num_samples if args.num_samples != 0 else 1000000
    num_epochs = args.epochs
    lr = args.lr
    training_method = args.task
    params = {"n": args.n, "alpha": 1e-6}

    print("Training method: {}".format(training_method))

    train_loader, validate_loader, test_loader = generateDataset(sp500_data, n=n, num_samples=num_samples, batch_size=args.batch_size)
    # feature_size = train_loader.dataset[0][0].shape[1]
    feature_size = train_loader.dataset[0][0].shape[1] * params["n"]
    params["x_dim"] = feature_size

    device = f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu'
    base_save = f'stock_portfolio_results_{args.n}_2'
    if not os.path.exists(base_save):
        os.makedirs(base_save)
    # if training_method != 'diffusion':
    #     model = PortfolioModel(input_size=feature_size, output_size=1)
    #     covariance_model = CovarianceModel(n=n)

    #     optimizer = torch.optim.Adam(list(model.parameters()) + list(covariance_model.parameters()), lr=lr)
    #     scheduler = ReduceLROnPlateau(optimizer, 'min')
    for run in range(args.seed, args.nRuns + args.seed):
        print(f"Run {run}, seed {run}")
        SEED = run
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        torch.backends.cudnn.deterministic = True

        save_folder = os.path.join(base_save, f"{run}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # ==================== Model initialization ====================
        if "diffusion" in training_method:
            diffusion_timesteps = 1000
            model = Diffsion(x_dim=feature_size, y_dim=params["n"], device=device, timesteps=diffusion_timesteps)
            optimizer = torch.optim.Adam(model.model_net.parameters(), lr=lr)
            pretrain_optimizer = torch.optim.Adam(model.model_net.parameters(), lr=1e-3)
            pretrain_loader, pretrain_validate_loader, pretrain_test_loader = generateDataset(sp500_data, n=n, num_samples=num_samples, batch_size=512)
            if args.pretrain_epochs > 0:
                model.pretrain_diffusion(pretrain_loader, pretrain_validate_loader, pretrain_optimizer, base_save, args, device)
            layer = DiffusionCvxpyModule(model, params, args.mc_samples, distr_est=True)
        elif "gaussian" in training_method:
            print("Training Gaussian MLP")
            model = GaussianMLP(feature_size, [1024, 1024], params["n"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            pretrain_loader, pretrain_validate_loader, pretrain_test_loader = generateDataset(sp500_data, n=n, num_samples=num_samples, batch_size=512)
            if args.pretrain_epochs > 0:
                model = run_rmse_net(model, pretrain_loader, pretrain_validate_loader, pretrain_optimizer, base_save, args, device)
            if training_method == "task_net_gaussian_reparam":
                layer = MLPCvxpyModule(model, params, args.mc_samples)
            elif training_method == "task_net_gaussian_distr":
                layer = MLPCvxpyModule_distr(model, params, args.mc_samples)
            elif training_method == "two_stage_gaussian":
                layer = None
            else:
                raise ValueError('Not implemented')
        elif "deterministic" in training_method or "mlp" in training_method:
            print("Training Deterministic MLP")
            model = MLP(input_dim=feature_size, hidden_dim=1024, output_dim=params["n"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            pretrain_loader, pretrain_validate_loader, pretrain_test_loader = generateDataset(sp500_data, n=n, num_samples=num_samples, batch_size=512)
            if args.pretrain_epochs > 0:
                model = run_rmse_net(model, pretrain_loader, pretrain_validate_loader, pretrain_optimizer, base_save, args, device)
            layer = MLPCvxpyModule(model, params, args.mc_samples)
        else:
            raise ValueError('Not implemented')

        # ==================== Training ====================
        train_loss_list, train_obj_list = [], []
        test_loss_list,  test_obj_list  = [], []
        validate_loss_list,  validate_obj_list = [], []

        print('n: {}, lr: {}'.format(n,lr))
        print('Start training...')
        evaluate = False
        total_forward_time, total_inference_time, total_qp_time, total_backward_time = 0, 0, 0, 0
        forward_time_list, inference_time_list, qp_time_list, backward_time_list = [], [], [], []

        best_val_obj = - float('inf')
        best_epoch = -1
        best_state_dict = None
     
        if 'two_stage' in training_method:
            if "diffusion" in training_method:
                eval_diffusion("two_stage_diffusion", model, train_loader, test_loader, params, args.mc_samples, save_folder)
                # torch.save(model.model_net.state_dict(), f"{save_folder}/{training_method}-two_stage_diffusion-SEED{SEED}.pth")
            elif "mlp" in training_method or "gaussian" in training_method:
                eval_net(training_method, model, train_loader, test_loader, params, save_folder, args.mc_samples)
                # torch.save(model.state_dict(), f"{save_folder}/{training_method}-SEED{SEED}.pth")
            continue        
        
        for epoch in range(-1, num_epochs):
            start_time = time.time()
            forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
            if 'deterministic' in training_method:
                if epoch == -1:
                    # print('Testing the optimal solution...')
                    # test_loss, test_obj = test_portfolio_mlp(model, layer, epoch, train_loader, params, device=device, evaluate=True)
                    # sys.stdout.write(f'Epoch {epoch} | Test Loss:     \t {test_loss:.8f} \t | Test Objective Value:     \t {test_obj*100:.6f}% \n')
                    continue
                else:
                    train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = train_portfolio_mlp(model, layer, optimizer, epoch, train_loader, params, device=device)
            elif "gaussian" in training_method:
                if epoch == -1:
                    # print('Testing the optimal solution...')
                    # test_loss, test_obj = test_portfolio_mlp(model, layer, epoch, train_loader, params, device=device, evaluate=True)
                    # sys.stdout.write(f'Epoch {epoch} | Test Loss:     \t {test_loss:.8f} \t | Test Objective Value:     \t {test_obj*100:.6f}% \n')
                    continue
                else:
                    train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = train_portfolio_mlp(model, layer, optimizer, epoch, train_loader, params, device=device)
            elif 'diffusion' in training_method:
                if epoch == -1:
                    # print('Testing the optimal solution...')
                    # test_loss, test_obj = test_portfolio_diffusion(model, layer, epoch, train_loader, params, device=device, evaluate=True)
                    # sys.stdout.write(f'Epoch {epoch} | Test Loss:     \t {test_loss:.8f} \t | Test Objective Value:     \t {test_obj*100:.6f}% \n')
                    continue
                else:
                    train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = train_portfolio_diffusion(model, layer, optimizer, epoch, train_loader, params, device=device)
            else:
                raise ValueError('Not implemented')
            
            total_forward_time   += forward_time
            total_inference_time += inference_time
            total_qp_time        += qp_time
            total_backward_time  += backward_time

            forward_time_list.append(forward_time)
            inference_time_list.append(inference_time)
            qp_time_list.append(qp_time)
            backward_time_list.append(backward_time)

            # ================ validating ==================
            if 'diffusion' in training_method:
                validate_loss, validate_obj = validate_portfolio_diffusion(model, layer, epoch, validate_loader, params, evaluate=evaluate, device=device)
            elif "mlp" in training_method or "gaussian" in training_method:
                validate_loss, validate_obj = validate_portfolio_mlp(model, layer, epoch, validate_loader, params, evaluate=evaluate, device=device)

            # ================== testing ===================
            if 'diffusion' in training_method:
                test_loss, test_obj = test_portfolio_diffusion(model, layer, epoch, test_loader, params, evaluate=evaluate, device=device)
            elif 'mlp' in training_method or "gaussian" in training_method:
                test_loss, test_obj = test_portfolio_mlp(model, layer, epoch, test_loader, params, evaluate=evaluate, device=device)

            # =============== printing data ================
            sys.stdout.write(f'Epoch {epoch} | Train Loss:    \t {train_loss:.8f} \t | Train Objective Value:    \t {train_obj*100:.6f}% \n')
            sys.stdout.write(f'Epoch {epoch} | Validate Loss: \t {validate_loss:.8f} \t | Validate Objective Value: \t {validate_obj*100:.6f}% \n')
            sys.stdout.write(f'Epoch {epoch} | Test Loss:     \t {test_loss:.8f} \t | Test Objective Value:     \t {test_obj*100:.6f}% \n')
            sys.stdout.flush()
            wandb.log({
                "train_obj": train_obj,
                "validate_obj": validate_obj,
                "test_obj": test_obj
            })

            if validate_obj > best_val_obj:
                best_val_obj = validate_obj
                if "diffusion" in training_method:
                    best_state_dict = copy.deepcopy(model.model_net.state_dict())
                    best_epoch = epoch
                    print(f"Best epoch: {best_epoch}")
                else:
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    print(f"Best epoch: {best_epoch}")

            # ============== recording data ================
            end_time = time.time()
            print("Epoch {}, elapsed time: {}, forward time: {}, inference time: {}, qp time: {}, backward time: {}".format(epoch, end_time - start_time, forward_time, inference_time, qp_time, backward_time))

            train_loss_list.append(train_loss)
            train_obj_list.append(train_obj)
            test_loss_list.append(test_loss)
            test_obj_list.append(test_obj)
            validate_loss_list.append(validate_loss)
            validate_obj_list.append(validate_obj)

            # record the data every epoch
            if not os.path.exists('results/performance/'):
                os.makedirs('results/performance/')
            if not os.path.exists('results/time/'):
                os.makedirs('results/time/')
            f_output = open('results/performance/' + filepath + "{}-SEED{}.csv".format(training_method,SEED), 'w')
            f_output.write('Epoch, {}\n'.format(epoch))
            f_output.write('training loss,' + ','.join([str(x) for x in train_loss_list]) + '\n')
            f_output.write('training obj,'  + ','.join([str(x) for x in train_obj_list])  + '\n')
            f_output.write('validating loss,' + ','.join([str(x) for x in validate_loss_list]) + '\n')
            f_output.write('validating obj,'  + ','.join([str(x) for x in validate_obj_list])  + '\n')
            f_output.write('testing loss,'  + ','.join([str(x) for x in test_loss_list])  + '\n')
            f_output.write('testing obj,'   + ','.join([str(x) for x in test_obj_list])   + '\n')
            f_output.close()

            f_time = open('results/time/' + filepath + "{}-SEED{}.csv".format(training_method, SEED), 'w')
            f_time.write('Epoch, {}\n'.format(epoch))
            f_time.write('Random seed, {}, forward time, {}, inference time, {}, qp time, {}, backward_time, {}\n'.format(str(seed), total_forward_time, total_inference_time, total_qp_time, total_backward_time))
            f_time.write('forward time,'   + ','.join([str(x) for x in forward_time_list]) + '\n')
            f_time.write('inference time,' + ','.join([str(x) for x in inference_time_list]) + '\n')
            f_time.write('qp time,'        + ','.join([str(x) for x in qp_time_list]) + '\n')
            f_time.write('backward time,'  + ','.join([str(x) for x in backward_time_list]) + '\n')
            f_time.close()

            # ============= early stopping criteria =============
            # kk = 10
            # if epoch >= kk*2-1:
            #     if training_method == 'two-stage':
            #         if evaluate:
            #             break
            #         GE_counts = np.sum(np.array(validate_loss_list[-kk:]) >= np.array(validate_loss_list[-2*kk:-kk]) - 1e-6)
            #         print('Generalization error increases counts: {}'.format(GE_counts))
            #         if GE_counts == kk or np.sum(np.isnan(validate_loss_list[-kk:])) == kk:
            #             evaluate = True
            #     else: # surrogate or decision-focused
            #         GE_counts = np.sum(np.array(validate_obj_list[-kk:]) <= np.array(validate_obj_list[-2*kk:-kk]) + 1e-6)
            #         print('Generalization error increases counts: {}'.format(GE_counts))
            #         if GE_counts == kk or np.sum(np.isnan(validate_obj_list[-kk:])) == kk:
            #             break
        
        if "diffusion" in training_method:
            model.model_net.load_state_dict(best_state_dict)
            if "two_stage" in training_method:
                eval_diffusion("two_stage_diffusion", model, train_loader, test_loader, params, args.mc_samples, save_folder)
                torch.save(model.model_net.state_dict(), f"{save_folder}/{training_method}-two_stage_diffusion-SEED{SEED}.pth")
                torch.save(test_obj, f"{save_folder}/{training_method}-two_stage_diffusion-SEED{SEED}")
            elif args.use_distr_est:
                eval_diffusion("diffusion_distr_est", model, train_loader, test_loader, params, args.mc_samples, save_folder)
                torch.save(model.model_net.state_dict(), f"{save_folder}/{training_method}-distr_est-SEED{SEED}.pth")
                torch.save(test_obj, f"{save_folder}/{training_method}-distr_est-SEED{SEED}")
            else:
                eval_diffusion("diffusion_point_est", model, train_loader, test_loader, params, args.mc_samples, save_folder)
                torch.save(model.model_net.state_dict(), f"{save_folder}/{training_method}-point_est-SEED{SEED}.pth")
                torch.save(test_obj, f"{save_folder}/{training_method}-point_est-SEED{SEED}")
        else:
            model.load_state_dict(best_state_dict)
            eval_net(training_method, model, train_loader, test_loader, params, save_folder, args.mc_samples)
            torch.save(model.state_dict(), f"{save_folder}/{training_method}-SEED{SEED}.pth")
            torch.save(test_obj, f"{save_folder}/{training_method}-SEED{SEED}")