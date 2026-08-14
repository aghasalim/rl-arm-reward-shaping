"""Training curves.

Plots task success rate, not training reward. Reward is not comparable across
reward versions -- that is the entire premise of the project -- so putting four
reward curves on one axis would be a chart that looks informative and means
nothing. Success rate is measured identically for every version.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "artifacts" / "logs"
FIGURES = ROOT / "reports" / "figures"

from .env import REWARD_VERSIONS  # noqa: E402

# Grey for the version that learns nothing, warm colours for the two the agent
# exploited, cool for the two that work.
COLORS = {"v1_sparse": "#999999", "v2_distance": "#d62728",
          "v3_penalties": "#ff7f0e", "v4_potential": "#9467bd",
          "v5_progress": "#1f77b4", "v6_goalfocus": "#2ca02c"}


def _binned(df: pd.DataFrame, col: str, bins: int = 60):
    """Bin episodes by timestep and average. Episodes finish at irregular
    timesteps across 8 parallel envs, so a fixed-width rolling window over the
    raw rows would weight late, short episodes more heavily."""
    edges = np.linspace(0, df["timestep"].max(), bins + 1)
    idx = np.digitize(df["timestep"], edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    g = df.groupby(idx)[col].mean()
    x = (edges[:-1] + edges[1:]) / 2
    return x[g.index.values], g.values


def shaping_comparison(out: Path | None = None) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for version, color in COLORS.items():
        f = LOGS / f"{version}_seed0.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        # Success is deliberately NOT one of these panels: at this 1.2M budget
        # every version scores a flat zero, so the panel would carry no
        # information while looking like a broken axis. Final distance is the
        # version-independent measure that actually separates them, and the
        # success numbers live in reports/results.md.
        for ax, col, label in zip(
            axes, ["final_dist", "collision", "ep_len"],
            ["mean final distance (m)", "collision rate", "episode length"],
        ):
            x, y = _binned(df, col)
            ax.plot(x / 1e6, y, color=color, label=version, lw=1.6)
            ax.set_xlabel("million timesteps")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
    axes[0].set_title("How close it gets (lower is better)")
    axes[1].set_title("Collision rate — exploit #1 is the pair pinned at 1.0")
    axes[2].set_title("Episode length — suicide vs stalling")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    out = Path(out or FIGURES / "reward_shaping_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"  -> {out}")
    return out


def multiseed(version: str | None = None, out: Path | None = None) -> Path:
    """version=None auto-detects the newest reward version that has a final run,
    so this does not silently plot an empty figure every time the reward improves."""
    if version is None:
        version = next(
            (v for v in reversed(REWARD_VERSIONS) if any(LOGS.glob(f"final_{v}_seed*.csv"))),
            REWARD_VERSIONS[-1],
        )
    files = sorted(LOGS.glob(f"final_{version}_seed*.csv")) or \
            sorted(LOGS.glob(f"{version}_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-seed logs for {version}; run `make final` first")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    curves = []
    for f in files:
        df = pd.read_csv(f)
        x, y = _binned(df, "success")
        ax.plot(x / 1e6, y, alpha=0.35, lw=1.2, color="#2ca02c")
        curves.append((x, y))
    if curves:
        n = min(len(y) for _, y in curves)
        Y = np.vstack([y[:n] for _, y in curves])
        X = curves[0][0][:n]
        mean, std = Y.mean(0), Y.std(0)
        ax.plot(X / 1e6, mean, color="#146b14", lw=2.4, label=f"mean of {len(curves)} seeds")
        ax.fill_between(X / 1e6, mean - std, mean + std, color="#2ca02c", alpha=0.18,
                        label="±1 std")
    ax.set_xlabel("million timesteps")
    ax.set_ylabel("success rate")
    ax.set_title(f"{version}: per-seed spread")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = Path(out or FIGURES / "multiseed.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"  -> {out}")
    return out


if __name__ == "__main__":
    shaping_comparison()
    multiseed()
