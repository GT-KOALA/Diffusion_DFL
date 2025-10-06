#/usr/bin/env python3

import os
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import warnings

warnings.filterwarnings("ignore")

def load_results(load_folders):
    load_folders = list(load_folders)
    array_files = ['diffusion_distr_est_resample_test_rmse', 'diffusion_point_est_test_rmse', 'task_net_mlp_gaussian_reparam_test_rmse', 'task_net_deterministic_mlp_test_rmse']
    float_tensor_files = ['diffusion_distr_est_resample_test_task', 'diffusion_point_est_test_task', 'task_net_mlp_gaussian_reparam_test_task', 'task_net_deterministic_mlp_test_task']
    col_names = ['Score fn', 'Reparam', 'Gaussian', 'Deterministic']

    df_rmse = pd.DataFrame()
    df_task = pd.DataFrame()
        
    for run_idx, folder in enumerate(load_folders):
        arrays, tensors = [], []
            
        for filename in array_files:
            with open(os.path.join(folder, filename), 'rb') as f:
                _res = np.load(f)
                if _res.size == 1:
                    raise ValueError("RMSE is 1-dim")
                arrays.append(_res)
                
        df = pd.DataFrame(pd.DataFrame(arrays).T)
        df.columns = col_names
        df['run'] = run_idx
        df_rmse = pd.concat([df_rmse, df], ignore_index=True)

        for filename in float_tensor_files:
            tensors.append(torch.load(os.path.join(folder, filename)))
        
        df = pd.DataFrame(pd.DataFrame(tensors).T)
        df.columns = col_names
        df['run'] = run_idx
        df_task = pd.concat([df_task, df], ignore_index=True)

    return df_rmse, df_task

def get_means_stds(df):
    # return df.groupby(df.index).mean(), df.groupby(df.index).std()
    df = df.copy()
    metric_cols = [c for c in df.columns if c != 'run']
    df['horizon'] = df.groupby('run').cumcount()

    # Aggregate across runs at each horizon
    means = df.groupby('horizon')[metric_cols].mean()
    stds  = df.groupby('horizon')[metric_cols].std(ddof=1)
    
    return means, stds

def plot_results(load_folders, save_folder):
    df_rmse, df_task = load_results(load_folders)
    rmse_mean, rmse_stds = get_means_stds(df_rmse)
    task_mean, task_stds = get_means_stds(df_task)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8.5, 3)

    styles = ['-', '--', '-.', '-.']
    colors = [sns.color_palette()[i] for i in [1,2,3]] + ['gray']

    ax = axes[0]
    # ax.set_axis_bgcolor('none')
    for col, style, color in zip(rmse_mean.columns, styles, colors):
        rmse_mean[col].plot(
            ax=ax, lw=2, fmt=style, color=color, yerr=rmse_stds[col])
    ax.set_ylabel('RMSE')

    ax2 = axes[1]
    # ax2.set_axis_bgcolor('none')
    for col, style, color in zip(task_mean.columns, styles, colors):
        if col == 'Cost-weighted RMSE':
            task_mean[col].plot(
                ax=ax2, lw=2, style=style, color=color)
            ax2.errorbar(task_mean.index+0.2, task_mean[col], 
                yerr=task_stds[col], color=color, lw=0, elinewidth=2)
        else:
            task_mean[col].plot(
                ax=ax2, lw=2, fmt=style, color=color, yerr=task_stds[col])
    ax2.set_ylabel('Task Loss')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    for a in [ax, ax2]:
        a.margins(0,0)
        a.grid(linestyle=':', linewidth='0.5', color='gray')
        a.xaxis.set_major_locator(ticker.MultipleLocator(4))
        a.set_xlim(0, 24)
        a.set_ylim(0, )

    # Joint x-axis label and legend
    fig.text(0.5, 0.13, 'Hour of Day', ha='center', fontsize=12)
    legend = ax.legend(loc='center left', bbox_to_anchor=(0.05, -0.4), 
        shadow=False, ncol=7, fontsize=12, borderpad=0, frameon=False)

    fig.savefig(os.path.join(save_folder, 
        'power_sched_24_hour.pdf'), dpi=500, format='pdf')

if __name__ == '__main__':
    YOUR_HOME_FOLDER = ""
    base_save = os.path.join(YOUR_HOME_FOLDER, "e2e-model-learning", "power_sched", "power_sched_results_final")
    nRuns = 10
    plot_results(map(
         lambda x: os.path.join(base_save, str(x)), range(nRuns)), base_save)