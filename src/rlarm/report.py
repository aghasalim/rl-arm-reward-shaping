"""Evaluate every trained policy against the one fixed success criterion and
write the results table.

Nothing here reads training reward. Each policy is re-run on the same 200 held-out
task layouts (seeds 10000-10199, disjoint from training) and scored identically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .env import GOAL_TOL, HOLD_STEPS, MAX_STEPS, REWARD_VERSIONS, VEL_TOL
from .evaluate import evaluate, format_row

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

KEYS = ["success_rate", "collision_rate", "timeout_rate", "mean_final_dist",
        "mean_len", "reached_target_rate", "settle_given_reached",
        "mean_speed_near", "mean_collision_step"]


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def _num(x, d=2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main(n_episodes: int = 200) -> dict:
    results: dict[str, dict] = {}

    print("random policy (floor):")
    results["random"] = evaluate(None, n_episodes=n_episodes)
    print("  " + format_row("random", results["random"]))

    print("\nreward-shaping versions (seed 0):")
    for v in REWARD_VERSIONS:
        f = ARTIFACTS / f"{v}_seed0.zip"
        if not f.exists():
            continue
        m = PPO.load(f, device="cpu")
        results[v] = evaluate(m, n_episodes=n_episodes, reward_version=v)
        print("  " + format_row(v, results[v]))

    # --- multi-seed final ---------------------------------------------------
    # Pick up whichever reward version the final run used, newest first, rather
    # than hard-coding a version that goes stale every time the reward improves.
    final_version = next(
        (v for v in reversed(REWARD_VERSIONS)
         if any("ckpt" not in f.name for f in ARTIFACTS.glob(f"final_{v}_seed*.zip"))),
        REWARD_VERSIONS[-1],
    )
    seeds = sorted(f for f in ARTIFACTS.glob(f"final_{final_version}_seed*.zip")
                   if "ckpt" not in f.name)
    print(f"\nfinal policy ({final_version}, {len(seeds)} seeds):")
    per_seed = []
    for f in seeds:
        m = PPO.load(f, device="cpu")
        # Must be the version these policies were trained on. Success and
        # collision are reward-independent, but mean_return is not: scoring a v6
        # policy under v4's reward reports a number from a reward function the
        # agent never saw.
        r = evaluate(m, n_episodes=n_episodes, reward_version=final_version)
        r["seed"] = int(f.stem.split("seed")[-1])
        per_seed.append(r)
        print("  " + format_row(f"final seed{r['seed']}", r))
    if per_seed:
        results["final_multiseed"] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([r[k] for r in per_seed])) for k in KEYS},
            "std": {k: float(np.std([r[k] for r in per_seed])) for k in KEYS},
            "n_seeds": len(per_seed),
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    _write_markdown(results, n_episodes)
    return results


def _write_markdown(results: dict, n_episodes: int) -> None:
    L = [
        "# Results",
        "",
        f"All policies evaluated on the same {n_episodes} held-out task layouts "
        "(seeds 10000+, disjoint from training), deterministic actions.",
        "",
        "**Success criterion, fixed before any training:** end effector within "
        f"{GOAL_TOL} m of the target, both joints below {VEL_TOL} rad/s, held for "
        f"{HOLD_STEPS} consecutive steps, no obstacle contact, within {MAX_STEPS} steps.",
        "",
        "## Reward versions",
        "",
        "| reward | success | collision | timeout | final dist (m) | ep len | reached target | settled \\| reached | speed in goal | mean collision step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = ["random"] + [v for v in REWARD_VERSIONS if v in results]
    for name in order:
        if name not in results:
            continue
        m = results[name]
        L.append(
            f"| {name} | {_pct(m['success_rate'])} | {_pct(m['collision_rate'])} | "
            f"{_pct(m['timeout_rate'])} | {_num(m['mean_final_dist'], 3)} | "
            f"{_num(m['mean_len'], 1)} | {_pct(m['reached_target_rate'])} | "
            f"{_pct(m['settle_given_reached'])} | {_num(m['mean_speed_near'])} | "
            f"{_num(m['mean_collision_step'], 1)} |"
        )

    if "final_multiseed" in results:
        ms = results["final_multiseed"]
        L += [
            "",
            f"## Final policy, {ms['n_seeds']} seeds",
            "",
            "| metric | mean | std | min | max |",
            "|---|---|---|---|---|",
        ]
        for k in ["success_rate", "collision_rate", "timeout_rate", "mean_final_dist", "mean_len"]:
            vals = [r[k] for r in ms["per_seed"]]
            fmt = _pct if k.endswith("_rate") else (lambda x: _num(x, 3))
            L.append(f"| {k} | {fmt(ms['mean'][k])} | {fmt(ms['std'][k])} | "
                     f"{fmt(min(vals))} | {fmt(max(vals))} |")
        L += ["", "Per-seed success rate: " + ", ".join(
            f"seed {r['seed']}: {r['success_rate']:.1%}" for r in ms["per_seed"]), ""]

    (REPORTS / "results.md").write_text("\n".join(L) + "\n")
    print(f"\n-> {REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
