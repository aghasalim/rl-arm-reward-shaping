"""PPO training with per-episode logging.

DummyVecEnv rather than SubprocVecEnv on purpose: a step of this environment
costs ~80 microseconds, which is below the process-IPC overhead, so subprocess
workers make it slower rather than faster. Measured before choosing.

No VecNormalize. Observations are already bounded and O(1) by construction (trig
encoding, normalized velocities), and normalizing the reward would rescale each
reward version differently -- precisely the confound this project is trying to
avoid. It also removes the classic failure of shipping a model without its
saved normalization statistics.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import ReachAvoidEnv

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
N_ENVS = 8


class EpisodeLogger(BaseCallback):
    """Records every finished episode. Success and collision come from the env's
    own info dict, so the log is in task terms and stays comparable across
    reward versions even though the reward column is not."""

    def __init__(self, out_csv: Path, checkpoint_at=(), model_stub: Path | None = None):
        super().__init__()
        self.out_csv = out_csv
        self.rows: list[dict] = []
        self.checkpoint_at = sorted(checkpoint_at)
        self.model_stub = model_stub
        self._next_ckpt = 0
        self._ep_ret = np.zeros(N_ENVS)
        self._ep_len = np.zeros(N_ENVS, dtype=int)

    def _on_step(self) -> bool:
        self._ep_ret += self.locals["rewards"]
        self._ep_len += 1
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            self.rows.append({
                "timestep": int(self.num_timesteps),
                "ep_return": float(self._ep_ret[i]),
                "ep_len": int(self._ep_len[i]),
                "success": int(bool(info.get("success", False))),
                "collision": int(bool(info.get("collision", False))),
                "final_dist": float(info.get("dist", float("nan"))),
            })
            self._ep_ret[i] = 0.0
            self._ep_len[i] = 0

        # Checkpoints exist so the showcase can show early/mid/late behaviour
        # from the same run rather than from three unrelated runs.
        while (self._next_ckpt < len(self.checkpoint_at)
               and self.num_timesteps >= self.checkpoint_at[self._next_ckpt]):
            if self.model_stub is not None:
                n = self.checkpoint_at[self._next_ckpt]
                self.model.save(f"{self.model_stub}_ckpt{n}")
            self._next_ckpt += 1
        return True

    def _on_training_end(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def make_env(reward_version: str, seed: int, rank: int, env_kwargs: dict | None = None):
    def _init():
        env = ReachAvoidEnv(reward_version=reward_version, **(env_kwargs or {}))
        env.reset(seed=seed * 1000 + rank)
        return env
    return _init


def train(reward_version: str, seed: int, timesteps: int,
          checkpoints=(), tag: str = "", env_kwargs: dict | None = None) -> Path:
    stub = ARTIFACTS / f"{tag or reward_version}_seed{seed}"
    stub.parent.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv(
        [make_env(reward_version, seed, i, env_kwargs) for i in range(N_ENVS)]
    )
    model = PPO(
        "MlpPolicy", venv, seed=seed, verbose=0,
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        # Linear decay to zero. Fine terminal positioning is the hard half of
        # this task and a constant 3e-4 keeps taking steps too large to settle
        # inside the 0.1 rad/s tolerance; `progress` runs 1 -> 0 in SB3.
        learning_rate=lambda progress: 3e-4 * progress, ent_coef=0.0,
        policy_kwargs={"net_arch": [128, 128]},
    )
    cb = EpisodeLogger(ARTIFACTS / "logs" / f"{stub.name}.csv",
                       checkpoint_at=checkpoints, model_stub=stub)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    model.save(stub)
    print(f"  {stub.name}: {timesteps:,} steps in {time.time() - t0:.0f}s "
          f"({timesteps / (time.time() - t0):,.0f} fps) -> {stub}.zip")
    return Path(f"{stub}.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reward-version", default="v6_goalfocus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--checkpoints", type=int, nargs="*", default=[])
    p.add_argument("--tag", default="", help="artifact name prefix; defaults to the reward version")
    args = p.parse_args()
    train(args.reward_version, args.seed, args.timesteps, args.checkpoints, tag=args.tag)
