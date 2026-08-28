"""Balance sweep for v5.

v5 removed the two exploits but still solved nothing: 0% success, and the arm
only reached the target neighbourhood in 6% of episodes. The suspicion is that
the collision penalty (-30) swamps the total achievable progress signal
(shaping_weight x initial distance ~ 7.5), so the risk-free policy is to barely
move.

The no-obstacle arm of this sweep is the load-bearing one. It separates "the
agent cannot learn to reach" from "the agent can reach but will not risk the
obstacle", which are indistinguishable from the success rate alone.

Run: python -m src.rlarm.sweep
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

ARMS = {
    # name:        (shaping_weight, collision_penalty, vel_weight, use_obstacle)
    # No dots in arm names: SB3 derives the save suffix with Path.suffix, and
    # "sweep_E_k0.5_seed0" has suffix ".5_seed0", which mangles the filename.
    "N_k000":      (10.0, 10.0, 0.00, True),
    "P_k005":      (10.0, 10.0, 0.05, True),
    "Q_k010":      (10.0, 10.0, 0.10, True),
}
TIMESTEPS = 4_000_000


def _run(item):
    name, (w, c, k, obs) = item
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    from stable_baselines3 import PPO

    from .evaluate import evaluate
    from .train import train

    kw = {"shaping_weight": w, "collision_penalty": c,
          "vel_weight": k, "use_obstacle": obs}
    train("v5_progress", seed=0, timesteps=TIMESTEPS, tag=f"sweep_{name}", env_kwargs=kw)
    model = PPO.load(f"artifacts/sweep_{name}_seed0.zip", device="cpu")
    # Always evaluated WITH the obstacle and the standard criterion, including
    # arm B: a policy that ignores obstacles must be scored against them.
    m = evaluate(model, n_episodes=150, reward_version="v5_progress")
    m["arm"] = name
    m["config"] = kw
    return m


def main():
    with mp.Pool(len(ARMS)) as pool:
        results = pool.map(_run, list(ARMS.items()))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'arm':<18} {'succ':>7} {'coll':>7} {'reached':>8} {'settle|r':>9} {'d_final':>8}")
    for m in sorted(results, key=lambda r: -r["success_rate"]):
        print(f"{m['arm']:<18} {m['success_rate']:7.1%} {m['collision_rate']:7.1%} "
              f"{m['reached_target_rate']:8.1%} {m['settle_given_reached']:9.1%} "
              f"{m['mean_final_dist']:8.3f}")


if __name__ == "__main__":
    main()
