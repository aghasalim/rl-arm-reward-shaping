"""A hand-written PD controller used as a feasibility oracle.

This is not a baseline to beat for its own sake. It answers a question that no
reward curve can: *is the success criterion physically achievable at all?* It
has exact inverse kinematics, so it already knows the answer; if it still cannot
satisfy the criterion, the environment is under-actuated and no reward function
will save it.

It ignores the obstacle entirely, which is deliberate, that is the gap the
learned policy has to justify itself against.

Run: python -m src.rlarm.oracle
"""
from __future__ import annotations

import math

import numpy as np

from .env import L1, L2, ReachAvoidEnv
from .evaluate import EVAL_SEED_BASE

KP, KD = 20.0, 6.0  # best of a small hand sweep; see NOTES.md


def inverse_kinematics(x: float, y: float, elbow_up: bool = True) -> tuple[float, float]:
    """Closed-form 2-link IK. Returns (q1, q2); unreachable points are clamped
    to the workspace boundary by the arccos clip rather than raising."""
    c2 = np.clip((x * x + y * y - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    s2 = math.sqrt(max(0.0, 1.0 - c2 * c2)) * (1.0 if elbow_up else -1.0)
    q2 = math.atan2(s2, c2)
    q1 = math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2)
    return q1, q2


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(n_episodes: int = 200, use_obstacle: bool = True,
        max_torque: float = 8.0, kp: float = KP, kd: float = KD) -> dict:
    env = ReachAvoidEnv(use_obstacle=use_obstacle, max_torque=max_torque)
    n_succ = n_coll = 0
    for i in range(n_episodes):
        env.reset(seed=EVAL_SEED_BASE + i)
        # Pick whichever elbow configuration is closer to the current pose.
        q = env.state[:2]
        desired = min(
            (inverse_kinematics(*env.target, True), inverse_kinematics(*env.target, False)),
            key=lambda d: abs(_wrap(d[0] - q[0])) + abs(_wrap(d[1] - q[1])),
        )
        done = False
        while not done:
            err = np.array([_wrap(desired[0] - env.state[0]),
                            _wrap(desired[1] - env.state[1])])
            action = np.clip((kp * err - kd * env.state[2:]) / env.max_torque, -1, 1)
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
        n_succ += bool(info["success"])
        n_coll += bool(info["collision"])
    return {"success_rate": n_succ / n_episodes, "collision_rate": n_coll / n_episodes,
            "max_torque": max_torque, "use_obstacle": use_obstacle}


def torque_sweep(torques=(2.0, 4.0, 6.0, 8.0, 12.0), n_episodes: int = 200) -> list[dict]:
    rows = []
    print(f"{'max_torque':>11} {'success (obstacle)':>19} {'collision':>10} "
          f"{'success (no obstacle)':>22}")
    for t in torques:
        withobs = run(n_episodes, True, t)
        noobs = run(n_episodes, False, t)
        rows.append({"max_torque": t, "with_obstacle": withobs, "no_obstacle": noobs})
        print(f"{t:>11.1f} {withobs['success_rate']:>19.1%} "
              f"{withobs['collision_rate']:>10.1%} {noobs['success_rate']:>22.1%}")
    return rows


if __name__ == "__main__":
    torque_sweep()
