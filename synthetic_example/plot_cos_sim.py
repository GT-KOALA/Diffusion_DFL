import re
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Optional

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


COS_RE = re.compile(
    r"cos_dp_10\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cos_dp_50\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cos_dp_100\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cos_dp_500\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"cos_rp\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)

EPOCH_RE = re.compile(r"Epoch\s*=\s*(\d+)")

METRICS = ["cos_dp_10", "cos_dp_50", "cos_dp_100", "cos_dp_500", "cos_rp"]


def _empty_sums_counts() -> Dict[str, Tuple[float, int]]:
    return {m: (0.0, 0) for m in METRICS}


def extract_epoch_cos_means(log_path: str) -> pd.DataFrame:
    sums_counts: Dict[Tuple[int, int], Dict[str, Tuple[float, int]]] = defaultdict(_empty_sums_counts)
    pending: List[Tuple[float, float, float, float, float]] = []

    seed_idx = 0
    last_epoch_seen: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Capture cosine lines
            m_cos = COS_RE.search(line)
            if m_cos:
                vals = tuple(float(m_cos.group(i)) for i in range(1, 6))
                pending.append(vals)
                continue

            # Capture epoch lines and attach any pending cos records to this epoch
            m_ep = EPOCH_RE.search(line)
            if m_ep:
                epoch = int(m_ep.group(1))

                # Detect new seed when epoch number decreases
                if last_epoch_seen is not None and epoch < last_epoch_seen:
                    seed_idx += 1
                last_epoch_seen = epoch

                if pending:
                    key = (seed_idx, epoch)
                    # accumulate all pending records to this epoch
                    # sums_counts dict stores (sum, count) per metric
                    current = sums_counts[key]
                    # To mutate entries, we'll reconstruct dict entries with new tuples
                    sums = {m: current[m][0] for m in METRICS}
                    counts = {m: current[m][1] for m in METRICS}

                    for (v10, v50, v100, v500, vrp) in pending:
                        for m, v in zip(METRICS, [v10, v50, v100, v500, vrp]):
                            if v != 0.0:
                                sums[m] += v
                                counts[m] += 1

                    # write back
                    for m in METRICS:
                        current_sum, current_cnt = sums[m], counts[m]
                        sums_counts[key][m] = (current_sum, current_cnt)

                    pending.clear()

    # Build DataFrame
    rows = []
    for (seed, epoch), sc in sorted(sums_counts.items(), key=lambda x: (x[0][0], x[0][1])):
        means = {}
        for m in METRICS:
            s, c = sc[m]
            means[m] = (s / c) if c > 0 else np.nan
        rows.append({"seed": seed, "epoch": epoch, **means, "log": os.path.basename(log_path)})

    df = pd.DataFrame(rows).sort_values(["log", "seed", "epoch"]).reset_index(drop=True)
    return df


def extract_from_many(log_paths: Iterable[str]) -> pd.DataFrame:
    frames = [extract_epoch_cos_means(p) for p in log_paths]
    if not frames:
        return pd.DataFrame(columns=["log", "seed", "epoch"] + METRICS)
    return pd.concat(frames, ignore_index=True).sort_values(["log", "seed", "epoch"]).reset_index(drop=True)


def aggregate_epoch_means_across_seeds(df: pd.DataFrame,
                                       metrics=METRICS,
                                       num_epochs: int = 10) -> pd.DataFrame:
    df_use = df[df["epoch"].between(0, num_epochs - 1)]
    agg = df_use.groupby("epoch")[list(metrics)].mean().reset_index()
    return agg


def plot_epoch_means(agg_df: pd.DataFrame,
                     metrics=METRICS,
                     title="Cosine similarity (mean across seeds)",
                     save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    x = agg_df["epoch"].values
    for m in metrics:
        if m in agg_df.columns:
            y = agg_df[m].values
            ax.plot(x, y, marker="o", label=m)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean across seeds")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


paths = ["pretrain_epochs=200_lr=1e-4.log"]
df_all = extract_from_many(paths)
csv_path = "cosine_means_by_seed_epoch.csv"
df_all.to_csv(csv_path, index=False)
df_agg = aggregate_epoch_means_across_seeds(df_all, num_epochs=10)

plot_epoch_means(df_agg,
                 metrics=["cos_dp_10","cos_dp_50","cos_dp_100","cos_dp_500"],
                 title="Cosine similarity between reparm and score function",
                 save_path="grad_cos.pdf")