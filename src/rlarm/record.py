"""Record episodes as GIFs.

Deterministic actions and fixed episode seeds, so the "early vs late" comparison
in the showcase shows the same task layouts at every training stage. Otherwise a
later checkpoint could look better purely because it drew easier targets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from .env import ReachAvoidEnv

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"


def record(model=None, out: Path | str = "out.gif", episodes: int = 3,
           reward_version: str = "v6_goalfocus", seed_base: int = 10_000,
           fps: int = 30, stride: int = 2) -> Path:
    env = ReachAvoidEnv(reward_version=reward_version, render_mode="rgb_array")
    frames = []
    outcomes = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        k = 0
        while not done:
            if k % stride == 0:  # every other frame keeps GIFs small
                frames.append(env.render())
            action = (env.action_space.sample() if model is None
                      else model.predict(obs, deterministic=True)[0])
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            k += 1
        frames.append(env.render())
        outcomes.append("success" if info["success"]
                        else "collision" if info["collision"] else "timeout")
        frames.extend([frames[-1]] * 8)  # brief hold so the outcome is readable

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"  {out.name}: {len(frames)} frames, outcomes={outcomes}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--reward-version", default="v6_goalfocus")
    args = p.parse_args()

    m = None
    if args.model:
        from stable_baselines3 import PPO
        m = PPO.load(args.model)
    record(m, args.out, args.episodes, args.reward_version)
