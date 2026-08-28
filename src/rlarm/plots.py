"""Training curves and one animation, all read from the committed episode logs.

Nothing here re-runs an experiment. Every figure reads artifacts/logs/*.csv, so
a plot can never disagree with the numbers in reports/results.md.

The curves show task metrics, never training reward. Reward is not comparable
across reward versions, that is the entire premise of the project, so
putting six reward curves on one axis would be a chart that looks informative
and means nothing. Distance, collision and success are measured identically for
every version.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "artifacts" / "logs"
FIGURES = ROOT / "reports" / "figures"

from .env import REWARD_VERSIONS  # noqa: E402
from .style import PALETTE, titled  # noqa: E402

# Grey for the version that learns nothing, warm for the two the agent
# exploited, cool for the two that work. A version keeps its colour in every
# figure here, so v6 is green wherever the reader meets it.
COLORS = {
    "v1_sparse": PALETTE[5],
    "v2_distance": PALETTE[1],
    "v3_penalties": PALETTE[3],
    "v4_potential": PALETTE[4],
    "v5_progress": PALETTE[0],
    "v6_goalfocus": PALETTE[2],
}
LABELS = {
    "v1_sparse": "v1 sparse",
    "v2_distance": "v2 distance",
    "v3_penalties": "v3 penalties",
    "v4_potential": "v4 potential",
    "v5_progress": "v5 progress",
    "v6_goalfocus": "v6 goal-focus",
}

XLABEL = "training timesteps (millions)"


def _binned(df: pd.DataFrame, col: str, bins: int = 60, tmax: float | None = None):
    """Bin episodes by timestep and average.

    Episodes finish at irregular timesteps across 8 parallel envs, so a
    fixed-width rolling window over the raw rows would weight late, short
    episodes more heavily. Passing tmax bins two runs of different length onto
    the same grid, so bins hold a comparable number of episodes in both.
    Empty bins come back as NaN rather than being silently dropped.
    """
    tmax = float(tmax if tmax is not None else df["timestep"].max())
    edges = np.linspace(0.0, tmax, bins + 1)
    idx = np.clip(np.digitize(df["timestep"], edges) - 1, 0, bins - 1)
    g = df.groupby(idx)[col].mean()
    y = np.full(bins, np.nan)
    y[g.index.values] = g.values
    return (edges[:-1] + edges[1:]) / 2, y


def _family(pattern: str, col: str, bins: int, tmax: float):
    """Every seed of one run family, binned onto a shared grid."""
    files = sorted(LOGS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no logs matching {pattern}")
    curves = [_binned(pd.read_csv(f), col, bins, tmax)[1] for f in files]
    x, _ = _binned(pd.read_csv(files[0]), col, bins, tmax)
    Y = np.vstack(curves)
    return x, Y, np.nanmean(Y, axis=0)


# Panel title, subtitle, column and axis label. The title says what the panel
# shows the reader, not which columns are on which axis.
PANELS = (
    ("final_dist", "mean distance to target at episode end (m)",
     "Only v5 and v6 close the distance",
     "one seed per version, all six on a 1.2M-step budget"),
    ("collision", "collision rate (fraction of episodes)",
     "v2 and v3 crash in every episode",
     "a crash ends the episode and stops the distance cost"),
    ("ep_len", "episode length (steps)",
     "One exploit crashes, the other stalls",
     "v2 and v3 end in 30 steps, v4 waits out the clock"),
)


def shaping_comparison(out: Path | None = None) -> Path:
    """The six reward versions side by side at a shared 1.2M-step budget."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # Success is deliberately not one of these panels: at this budget every
    # version scores a flat zero, so the panel would carry no information while
    # looking like a broken axis. Distance is the version-independent measure
    # that actually separates them, and the success numbers are in results.md.
    for version, color in COLORS.items():
        f = LOGS / f"{version}_seed0.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for ax, (col, _, _, _) in zip(axes, PANELS):
            x, y = _binned(df, col)
            ax.plot(x / 1e6, y, color=color, label=LABELS[version], lw=1.7)

    for ax, (_, ylabel, title, subtitle) in zip(axes, PANELS):
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(ylabel)
        titled(ax, title, subtitle)

    # One legend for all three panels, under the row, so no line is ever hidden
    # behind a key.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(out or FIGURES / "reward_shaping_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def multiseed(version: str | None = None, out: Path | None = None) -> Path:
    """Per-seed training curves for the final reward.

    version=None auto-detects the newest reward version that has a final run,
    so this does not silently plot an empty figure every time the reward
    improves.
    """
    if version is None:
        version = next(
            (v for v in reversed(REWARD_VERSIONS) if any(LOGS.glob(f"final_{v}_seed*.csv"))),
            REWARD_VERSIONS[-1],
        )
    pattern = f"final_{version}_seed*.csv"
    if not any(LOGS.glob(pattern)):
        pattern = f"{version}_seed*.csv"
    if not any(LOGS.glob(pattern)):
        raise FileNotFoundError(f"no per-seed logs for {version}; run `make final` first")

    color = COLORS.get(version, PALETTE[2])
    tmax = max(pd.read_csv(f)["timestep"].max() for f in LOGS.glob(pattern))
    x, Y, mean = _family(pattern, "success", bins=60, tmax=tmax)
    n = Y.shape[0]
    std = np.nanstd(Y, axis=0)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for y in Y:
        ax.plot(x / 1e6, y, alpha=0.32, lw=1.1, color=color)
    ax.fill_between(x / 1e6, mean - std, mean + std, color=color, alpha=0.16,
                    linewidth=0, label="1 std across seeds")
    ax.plot(x / 1e6, mean, color=color, lw=2.4, label=f"mean of {n} seeds")

    # Guarded so the title cannot claim a takeoff that is not in the data: a
    # shorter budget, or a reward that never solves anything, gets the flat
    # statement instead.
    above = np.nan_to_num(mean) > 0.01
    first = x[int(np.argmax(above))] / 1e6 if above.any() else None
    if first is not None:
        ax.axvline(first, color="#777777", linestyle="--", lw=1.0, zorder=1)
        ax.annotate(f"first success near {first:.1f}M steps", xy=(first, 0.93),
                    xycoords=("data", "axes fraction"), xytext=(7, 0),
                    textcoords="offset points", fontsize=9, color="#555555", va="center")

    ax.set_xlabel(XLABEL)
    ax.set_ylabel("success rate (fraction of training episodes)")
    ax.set_ylim(bottom=0)
    titled(ax,
           f"Nothing happens for {first:.1f}M steps, then every seed learns at once"
           if first is not None else "No seed ever solves an episode at this budget",
           f"{LABELS.get(version, version)}, success on training episodes with "
           "stochastic actions; the held-out score is in results.md")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = Path(out or FIGURES / "multiseed.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _shrink_gif(path: Path) -> Path:
    """Rewrite every frame onto one shared palette. Roughly halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], loop=0,
              duration=durations, optimize=True)
    return path


def longer_training(out: Path | None = None, frames: int = 66, hold: int = 14,
                    fps: int = 12) -> Path:
    """The 3M and 8M runs of the same reward, drawn as training goes on.

    Both families are already in artifacts/logs, five seeds each, and the only
    thing that differed between them was --timesteps. The 8M curve peaks and
    then slides back, which is the result the README reports and the one a
    single end-of-run number cannot show.
    """
    bin_steps = 1.0e5  # same bin width for both families, so both are equally smooth
    x3, Y3, m3 = _family("final_v6_goalfocus_seed*.csv", "success",
                         bins=int(3.0e6 / bin_steps), tmax=3.0e6)
    x8, Y8, m8 = _family("long8m_seed*.csv", "success",
                         bins=int(8.0e6 / bin_steps), tmax=8.0e6)
    short, long_ = PALETTE[2], PALETTE[1]
    peak = int(np.nanargmax(m8))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, 8.3)
    ax.set_ylim(0, float(np.nanmax([Y3.max(), Y8.max()])) * 1.12)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("success rate (fraction of training episodes)")
    titled(ax, f"Training longer peaks near {x8[peak] / 1e6:.1f}M, then slides back down",
           "same reward and the same five seeds, only --timesteps changed")

    art = {}
    for key, X, Y, colour, label in (("s", x3, Y3, short, "stopped at 3M steps"),
                                     ("l", x8, Y8, long_, "run out to 8M steps")):
        art[key + "_seeds"] = [ax.plot([], [], color=colour, alpha=0.22, lw=0.9)[0]
                               for _ in Y]
        art[key] = ax.plot([], [], color=colour, lw=2.4, label=f"{label}, mean of {len(Y)}")[0]
        art[key + "_head"] = ax.plot([], [], "o", markersize=6.0, color=colour)[0]
    ax.legend(loc="lower right")

    art["vline"] = ax.axvline(3.0, color="#999999", linestyle="--", lw=1.0, visible=False)
    art["vtext"] = ax.text(3.08, 0.97, "", transform=ax.get_xaxis_transform(),
                           fontsize=9, color="#666666", ha="left", va="top")
    art["peak"] = ax.plot([], [], "v", markersize=8, color="#333333")[0]
    art["peaktext"] = ax.text(x8[peak] / 1e6, m8[peak], "", fontsize=9, color="#333333",
                              ha="center", va="bottom")
    art["readout"] = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9.5,
                             color="#555555", ha="left", va="top")
    # Footnote rather than an in-axes note: the bottom right of the plot is the
    # only empty corner and the legend already has it.
    fig.text(0.012, 0.014, "the learning rate decays to zero over the run, so the 8M job is "
                           "not the 3M job continued",
             fontsize=8.3, color="#8a8a8a", ha="left", va="bottom")

    cuts = np.linspace(0.0, 8.0e6, frames)

    def draw(i):
        cut = cuts[min(i, frames - 1)]
        for key, X, Y, m in (("s", x3, Y3, m3), ("l", x8, Y8, m8)):
            k = int(np.searchsorted(X, cut))
            for line, y in zip(art[key + "_seeds"], Y):
                line.set_data(X[:k] / 1e6, y[:k])
            art[key].set_data(X[:k] / 1e6, m[:k])
            art[key + "_head"].set_data(X[k - 1:k] / 1e6, m[k - 1:k])
        art["readout"].set_text(f"training step: {cut / 1e6:.1f}M")
        if cut >= 3.0e6:
            art["vline"].set_visible(True)
            art["vtext"].set_text("the short runs stopped here")
        if cut >= x8[peak]:
            art["peak"].set_data([x8[peak] / 1e6], [m8[peak] + 0.03])
            art["peaktext"].set_text(f"peak {m8[peak]:.0%} at {x8[peak] / 1e6:.1f}M")
            art["peaktext"].set_position((x8[peak] / 1e6, m8[peak] + 0.045))
        if i >= frames:
            art["readout"].set_text(
                f"8M ends at {np.nanmean(m8[-5:]):.0%}, below the {np.nanmean(m3[-5:]):.0%}\n"
                "the 3M runs were already at")
        return []

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    anim = FuncAnimation(fig, draw, frames=frames + hold, interval=1000 // fps)
    out = Path(out or FIGURES / "longer_training.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    # dpi is set here and nowhere else: the house style renders stills at 170,
    # which would put this GIF over the size a README should carry.
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return _shrink_gif(out)


def main() -> None:
    for p in (shaping_comparison(), multiseed(), longer_training()):
        print(f"  -> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
