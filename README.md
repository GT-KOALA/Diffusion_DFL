# Diffusion Decision-Focused Learning Implementations

This repo contains three self-contained experiments comparing **decision-focused learning** approaches (Diffusion models, Gaussian, and deterministic MLP) on synthetic and real optimization tasks. Each folder can be run independently.

## Environment

- Python 3.9+
- PyTorch (GPU recommended)
- NumPy, Pandas, TQDM
- Joblib (for parallelization, version >= 1.5.0)
- QPTH, CVXPY (and optionally CVXPyLayers for the portfolio task)
- scikit-learn (for power scheduling)
- Weights & Biases (optional; can be set to offline)

Quick install (example):

```bash
pip install torch numpy pandas tqdm joblib scikit-learn cvxpy qpth cvxpylayers wandb
```

**GPU selection**: pass `--cuda_device <id>` to scripts (or see each folder’s `constants.py`).

**Disable W&B (optional):**

```bash
export WANDB_MODE=offline
# or set WANDB_API_KEY in your environment if you want to log runs
```

---

## 1) `synthetic_example/`

Toy stochastic decision task to illustrate when **stochastic** training (diffusion / distributional) can outperform **deterministic** predictors under task loss. Includes a differentiable lower-level solver via KKT.

**Key files**

- `main_toy.py` — entrypoint for all methods
- `data_gen.py` — synthetic data generators (1D/2D, class-conditional setups)
- `cvxpy_toy_kkt.py` — KKT-based differentiable solver and task loss
- `model_classes.py`, `nets_toy.py` — model definitions & training loops
- `cvxpylayer_local/` — local CVXPY layer utilities

**Typical runs**

```bash
cd synthetic_example

# Diffusion (score-function estimator)
python main_toy.py --use_distr_est --task diffusion

# Diffusion (reparameterization estimator)
python main_toy.py --task diffusion

# Gaussian (score-function estimator)
python main_toy.py --task task_net_mlp_gaussian_distr

# Gaussian (reparameterization estimator)
python main_toy.py --task task_net_gaussian_reparam

# Deterministic MLP (determinstic optimization)
python main_toy.py --task task_net_deterministic_mlp

# Two-stages
python main_toy.py --task two_stage_diffusion &
python main_toy.py --task two_stage_mlp &
python main_toy.py --task two_stage_gaussian &

# Policy learning
python main_toy.py --task policy_learning &
```

Useful flags (subset):

- `--mc_samples` Monte-Carlo samples for distributional methods
- `--pretrain_epochs`, `--pretrain_model_path` optional diffusion pretraining
- `--seed`, `--interval` (replay buffer step), `--lr` learning rate

**Outputs & logging**

- W&B runs under `wandb/` (prefix `diffusion_distr_est`, `diffusion_point_est`, etc.)
- Local artifacts/results printed/saved by training scripts

---

## 2) `power_sched/`

Simplified **power grid scheduling** task (time-series features & decisions) with a differentiable lower-level program. Compares diffusion (distributional & point), Gaussian (reparam / score-function), and deterministic MLP.

**Key files**

- `main.py` — entrypoint (k-fold split support, holidays/features handling)
- `cvxpy_powers_sched(_kkt).py`, `cvxpy_qp.py` — differentiable solver wrappers
- `solver_distribution*.py`, `solver_reparam.py`, `solver_mlp.py` — method variants
- `nets.py`, `model_classes.py` — models and training/eval
- `qpthlocal/` — local QP solver (PDIPM) if used by some variants
- `util.py`, `batch.py` — dataset & loader utilities

**Typical runs**

```bash
cd power_sched

# Diffusion (distributional / score-function)
python main.py --task diffusion --use_distr_est --lr 8e-6

# Diffusion (reparameterization estimator)
python main.py --task diffusion --lr 1e-5 --mc_samples 30

# Gaussian (score-function estimator)
python main.py --task task_net_mlp_gaussian_distr

# Gaussian (reparameterization estimator)
python main.py --task task_net_gaussian_reparam

# Deterministic MLP (determinstic optimization)
python main.py --task task_net_deterministic_mlp

# Two-stages
python main.py --task two_stage_diffusion &
python main.py --task two_stage_mlp &
python main.py --task two_stage_gaussian &

# Policy learning
python main.py --task policy_learning &
```

Useful flags (subset):

- `--sf_sample_size` (score-function sampling), `--mc_samples`
- `--pretrain_epochs`, `--pretrain_model_path` (diffusion pretraining)
- `--batch_size`, `--seed`, `--lr`, `--cuda_device`

**Outputs & logging**

- W&B runs under `wandb/` (prefix `ps_*`)
- Local results (see `nets.py` and `plot.py` helpers for aggregations/plots)

---

## 3) `stock_portfolio/`

**Mean-variance portfolio optimization** with differentiable solver layers and diffusion/Gaussian/deterministic predictors. Data loader builds features/targets from historical prices (Quandl).

**Before you run**

- Set your Quandl API key in `stock_portfolio/key.py`:

  ```python
  API_KEY = "YOUR_KEY"
  ```

  (Used by `data_utils.py` via `quandl.ApiConfig.api_key`.)

**Key files**

- `main.py` — entrypoint for training/evaluation
- `data_utils.py` — downloads & prepares SP500-style data/features
- `cvxpy_stock_portfolio.py` — KKT-based differentiable portfolio layer
- `portfolio_utils*.py`, `solver_*.py` — method implementations
- `utils.py`, `sqrtm.py` — math helpers

**Typical runs**

```bash
cd stock_portfolio

# Diffusion (score-function)
python main.py --task diffusion --n 50 --epochs 50  --lr 5e-5

# Diffusion (reparameterization estimator)
python main.py --task diffusion --n 50 --epochs 50

# Gaussian (score-function estimator)
python main.py --task task_net_mlp_gaussian_distr --n 50 --epochs 50

# Gaussian (reparameterization estimator)
python main.py --task task_net_gaussian_reparam --n 50 --epochs 50

# Deterministic MLP (determinstic optimization)
python main.py --task task_net_deterministic_mlp --n 50 --epochs 50

# Two-stages
python main.py --task two_stage_diffusion &
python main.py --task two_stage_mlp &
python main.py --task two_stage_gaussian &

# Policy learning
python main.py --task policy_learning &
```

Useful flags (subset):

- `--n` number of securities (e.g., 50)
- `--epochs`, `--lr`, `--batch_size`
- `--mc_samples` for distributional methods
- `--T-size` for reparameterization variants
- `--num-samples` to cap dataset size
- `--filepath` to set an output filename prefix under results

**Outputs & logging**

- W&B runs under `wandb/` (prefix `stock_portfolio_*`)
- Results/CSV aggregations written by `main.py` and `process.py`
- Many scripts expect/write under folders like `stock_portfolio_results_*` (already covered by your `.gitignore`)

---

## Plot figures in paper

- Figure 1: plot_motivation.ipynb in ./synthetic_example
- Figure 2: plot_cos_sim.py in ./synthetic_example
- Figure 3: plot_compare_resample.ipynb in ./power_sched
- Figure 4: plot_compare_sf_samples_size.ipynb in ./power_sched
- Figure 5: plot_multi_portfolio.ipynb in ./stock_portfolio

## Tips & Troubleshooting

- If CVXPY complains about solvers, install one of `OSQP`, `ECOS`, or a commercial solver you have licenses for (and set `CVXPY` default accordingly).
- For reproducibility, use `--seed` and fix CUDA device with `--cuda_device`.
- On clusters, ensure CUDA visibility (`CUDA_VISIBLE_DEVICES`) matches `--cuda_device`.
