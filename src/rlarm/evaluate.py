"""Version-independent evaluation.

The whole point of this project is comparing reward functions, which makes
training reward useless as a comparison metric -- v2 and v4 are not even
measured in the same units, and "v4 scores higher" would be meaningless. So
every policy is judged against one fixed, external criterion defined before any
training happened:

    SUCCESS = end effector within 5 cm of the target, both joints below
              0.1 rad/s, held for 10 consecutive steps, without touching
              the obstacle, within 200 steps.

The extra diagnostics exist to catch reward hacking. A policy can look fine on
success rate alone while doing something ridiculous, so we also measure whether
it got close and then failed to stop (orbiting), and how quickly collision
episodes end (deliberate self-termination).
"""
from __future__ import annotations

import numpy as np

from .env import GOAL_TOL, MAX_STEPS, ReachAvoidEnv

NEAR = 0.15  # metres: "got to the neighbourhood of the target"
EVAL_SEED_BASE = 10_000  # disjoint from training seeds


def evaluate(model=None, n_episodes: int = 200, reward_version: str = "v6_goalfocus",
             deterministic: bool = True) -> dict:
    """Run n_episodes on fixed task layouts. model=None evaluates a random policy.

    Episode seeds are fixed and shared across every reward version, so all
    versions face the identical set of target/obstacle layouts. Without this the
    comparison would be confounded by task difficulty.
    """
    env = ReachAvoidEnv(reward_version=reward_version)
    n_succ = n_coll = n_near = n_near_succ = 0
    dists, lens, coll_lens, near_speeds, returns = [], [], [], [], []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        # reset(seed=) seeds the task layout but not the action space, so the
        # random-policy baseline moved by a few points between report runs and
        # the published table could not be reproduced exactly. Seed both.
        env.action_space.seed(EVAL_SEED_BASE + i)
        done = False
        ret = 0.0
        steps = 0
        min_dist = float("inf")
        speeds_near = []
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            steps += 1
            min_dist = min(min_dist, info["dist"])
            if info["dist"] < NEAR:
                speeds_near.append(float(np.abs(env.state[2:]).max()))
            done = term or trunc

        n_succ += bool(info["success"])
        n_coll += bool(info["collision"])
        got_near = min_dist < NEAR
        n_near += got_near
        n_near_succ += got_near and bool(info["success"])
        dists.append(info["dist"])
        lens.append(steps)
        returns.append(ret)
        if info["collision"]:
            coll_lens.append(steps)
        near_speeds.extend(speeds_near)

    return {
        "success_rate": n_succ / n_episodes,
        "collision_rate": n_coll / n_episodes,
        "timeout_rate": 1 - (n_succ + n_coll) / n_episodes,
        "mean_final_dist": float(np.mean(dists)),
        "mean_len": float(np.mean(lens)),
        "mean_return": float(np.mean(returns)),
        # --- reward-hacking diagnostics ---
        # reached_target_rate: got within 15 cm at any point.
        "reached_target_rate": n_near / n_episodes,
        # settle_given_reached: of those, how many actually stopped there. A low
        # value with a high reached rate is the signature of orbiting/overshoot.
        "settle_given_reached": (n_near_succ / n_near) if n_near else 0.0,
        # mean_speed_near: how fast it is still moving inside the goal region.
        "mean_speed_near": float(np.mean(near_speeds)) if near_speeds else float("nan"),
        # mean_collision_step: collisions arriving far earlier than the step
        # limit suggest the agent is ending episodes on purpose.
        "mean_collision_step": float(np.mean(coll_lens)) if coll_lens else float("nan"),
        "n_episodes": n_episodes,
    }


def format_row(name: str, m: dict) -> str:
    return (
        f"{name:<16} succ {m['success_rate']:6.1%}  coll {m['collision_rate']:6.1%}  "
        f"timeout {m['timeout_rate']:6.1%}  d_final {m['mean_final_dist']:.3f}  "
        f"len {m['mean_len']:5.1f}  reached {m['reached_target_rate']:6.1%}  "
        f"settle|reached {m['settle_given_reached']:6.1%}  "
        f"v_near {m['mean_speed_near']:.2f}  coll_step {m['mean_collision_step']:5.1f}"
    )


if __name__ == "__main__":
    m = evaluate(None, n_episodes=200)
    print(format_row("random", m))
    print(f"\nSuccess criterion: within {GOAL_TOL} m, both joints < 0.1 rad/s, "
          f"held 10 steps, no collision, within {MAX_STEPS} steps.")
