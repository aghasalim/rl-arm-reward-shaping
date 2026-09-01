"""Export the fixtures the other implementations in verify/ are checked against.

Nothing here checks anything. It dumps three things out of the Python that the
README's numbers come from, so that C, Go, Java, Node, R, Rust and SQL can
recompute them without importing any of it:

  physics_trace.csv   RK4 integrator output for four fixed initial states
  reward_trace.csv    every input to env._reward and the reward it returned
  oracle_layouts.csv  the 200 held-out task layouts, for both obstacle settings

Regenerate with:  .venv/bin/python verify/export_golden.py [output-dir]
Check instead:    .venv/bin/python verify/export_golden.py --check
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rlarm import env as E
from src.rlarm.env import ReachAvoidEnv
from src.rlarm.evaluate import EVAL_SEED_BASE
from src.rlarm.oracle import KD, KP, _wrap, inverse_kinematics

GOLDEN = Path(__file__).resolve().parent / "golden"
# --check compares a fresh dump against the committed fixtures instead of
# writing, which is how verify.sh notices that env.py has moved and the
# fixtures the other seven implementations read have not.
CHECK = "--check" in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
OUT = Path(ARGS[0]) if ARGS else GOLDEN

# Not byte equality: the layout coordinates go through numpy trigonometry and
# numpy.linalg.solve, and LAPACK gives the last ulp differently on a Linux
# runner than it does on this laptop. Every integer column, which includes every
# oracle outcome flag, still has to match exactly.
RTOL = 1e-9
TORQUES = (2.0, 4.0, 6.0, 8.0, 12.0)  # src/rlarm/oracle.py::torque_sweep
SUCCESS_SEED_OFFSET = 0  # seed 10000: the PD oracle settles there in 100 steps

# state, torque, damping, steps, log every
PHYSICS_CASES = [
    ("unforced_nodamp", [0.3, -0.8, 2.0, -1.5], [0.0, 0.0], 0.0, 500, 50),
    ("unforced_damped", [0.3, -0.8, 3.0, -2.0], [0.0, 0.0], 0.10, 200, 20),
    ("torque_step", [-1.2, 2.4, 0.5, 4.0], [8.0, -3.0], 0.10, 200, 20),
    ("torque_from_rest", [0.0, 0.0, 0.0, 0.0], [-2.5, 8.0], 0.10, 200, 20),
]


def export_physics() -> None:
    rows = []
    old = E.DAMPING
    try:
        for name, s0, tau, damping, steps, every in PHYSICS_CASES:
            E.DAMPING = damping
            env = ReachAvoidEnv()
            env.state = np.array(s0, dtype=np.float64)
            t = np.array(tau, dtype=np.float64)
            for step in range(steps + 1):
                if step:
                    env._integrate(t)
                if step % every == 0:
                    q1, q2, dq1, dq2 = env.state
                    rows.append({
                        "case": name, "damping": repr(damping),
                        "tau1": repr(tau[0]), "tau2": repr(tau[1]),
                        "step": step, "q1": repr(float(q1)), "q2": repr(float(q2)),
                        "dq1": repr(float(dq1)), "dq2": repr(float(dq2)),
                        "energy": repr(float(E.kinetic_energy(env.state))),
                    })
    finally:
        E.DAMPING = old
    _write(OUT / "physics_trace.csv", rows)


def export_reward() -> None:
    """Record the arguments of every _reward call and what it returned.

    The recomputation in verify/reward.js has to be able to reproduce the reward
    from the row alone, so the two pieces of hidden state the reward reads,
    _prev_dist for v4 and _prev_pot for v5 and v6, are captured before the call.
    """
    rows = []
    original = ReachAvoidEnv._reward

    def record(self, dist, tau, collided, success):
        prev_dist, prev_pot = self._prev_dist, self._prev_pot
        speed = float(np.linalg.norm(self.state[2:]))
        settled = self._settled()
        r = original(self, dist, tau, collided, success)
        rows.append({
            "version": self.reward_version, "episode": self._episode,
            "step": self._steps, "dist": repr(dist), "prev_dist": repr(prev_dist),
            "prev_pot": repr(prev_pot), "speed": repr(speed),
            "tau1": repr(float(tau[0])), "tau2": repr(float(tau[1])),
            "collided": int(collided), "settled": int(settled),
            "success": int(success),
            "shaping_weight": repr(self.shaping_weight),
            "collision_penalty": repr(self.collision_penalty),
            "vel_weight": repr(self.vel_weight), "time_cost": repr(self.time_cost),
            "goal_focus": repr(self.goal_focus if self.reward_version == "v6_goalfocus" else 0.0),
            "max_torque": repr(self.max_torque),
            "reward": repr(float(r)),
        })
        return r

    ReachAvoidEnv._reward = record
    try:
        for version in E.REWARD_VERSIONS:
            env = ReachAvoidEnv(reward_version=version)
            for ep in range(3):
                env._episode = ep
                env.reset(seed=EVAL_SEED_BASE + ep)
                # A fixed torque sequence, not a policy: the point is to visit
                # collisions, near misses and the step limit, not to be good.
                rng = np.random.default_rng(1234 + ep)
                done = False
                while not done:
                    a = np.clip(rng.normal(0.0, 0.6, size=2), -1, 1)
                    _, _, term, trunc, _ = env.step(a)
                    done = term or trunc
            # One episode driven by the PD oracle, so the settled and success
            # branches of the reward are in the trace too. Random torques never
            # reach either.
            env._episode = 3
            env.reset(seed=EVAL_SEED_BASE + SUCCESS_SEED_OFFSET)
            q = env.state[:2]
            desired = min(
                (inverse_kinematics(*env.target, True),
                 inverse_kinematics(*env.target, False)),
                key=lambda d: abs(_wrap(d[0] - q[0])) + abs(_wrap(d[1] - q[1])),
            )
            done = False
            while not done:
                err = np.array([_wrap(desired[0] - env.state[0]),
                                _wrap(desired[1] - env.state[1])])
                a = np.clip((KP * err - KD * env.state[2:]) / env.max_torque, -1, 1)
                _, _, term, trunc, _ = env.step(a)
                done = term or trunc
    finally:
        ReachAvoidEnv._reward = original
    _write(OUT / "reward_trace.csv", rows)


def export_oracle_layouts() -> None:
    """The 200 evaluation layouts, and what the PD oracle does on each one.

    use_obstacle changes the layout: reset resamples the start pose until it is
    collision free, so with the obstacle disabled the first draw is always kept
    and the random stream lands elsewhere.

    The per-episode outcome columns are what make verify/oracle a real check.
    The published torque table is a proportion over 200 episodes, so it only
    moves in steps of 0.5% and a single wrong layout can leave every rounded
    cell unchanged. Recording success and collision for each episode at each
    torque means the Rust replay has to agree 4000 times, not 10.
    """
    rows = []
    for use_obstacle in (True, False):
        env = ReachAvoidEnv(use_obstacle=use_obstacle)
        for i in range(200):
            env.reset(seed=EVAL_SEED_BASE + i)
            row = {
                "use_obstacle": int(use_obstacle), "seed": EVAL_SEED_BASE + i,
                "target_x": repr(float(env.target[0])), "target_y": repr(float(env.target[1])),
                "obstacle_x": repr(float(env.obstacle[0])), "obstacle_y": repr(float(env.obstacle[1])),
                "q1": repr(float(env.state[0])), "q2": repr(float(env.state[1])),
            }
            for t in TORQUES:
                ok, hit = _oracle_episode(use_obstacle, EVAL_SEED_BASE + i, t)
                row[f"success_t{t:g}"] = int(ok)
                row[f"collision_t{t:g}"] = int(hit)
            rows.append(row)
    _write(OUT / "oracle_layouts.csv", rows)


def _oracle_episode(use_obstacle: bool, seed: int, max_torque: float) -> tuple[bool, bool]:
    """One PD-oracle episode, the same loop src/rlarm/oracle.py::run drives."""
    env = ReachAvoidEnv(use_obstacle=use_obstacle, max_torque=max_torque)
    env.reset(seed=seed)
    q = env.state[:2]
    desired = min(
        (inverse_kinematics(*env.target, True), inverse_kinematics(*env.target, False)),
        key=lambda d: abs(_wrap(d[0] - q[0])) + abs(_wrap(d[1] - q[1])),
    )
    done = False
    while not done:
        err = np.array([_wrap(desired[0] - env.state[0]),
                        _wrap(desired[1] - env.state[1])])
        action = np.clip((KP * err - KD * env.state[2:]) / env.max_torque, -1, 1)
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
    return bool(info["success"]), bool(info["collision"])


def _write(path: Path, rows: list[dict]) -> None:
    if CHECK:
        _compare(GOLDEN / path.name, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{path.name:22} {len(rows):6d} rows")


FAILURES = []


def _compare(path: Path, rows: list[dict]) -> None:
    """Require the committed fixture to be the file this dump would write."""
    with path.open(newline="") as f:
        old = list(csv.DictReader(f))
    if not old:
        FAILURES.append(f"{path.name}: empty")
        return
    if list(old[0]) != list(rows[0]):
        FAILURES.append(f"{path.name}: columns are {list(old[0])}, dump has {list(rows[0])}")
        return
    if len(old) != len(rows):
        FAILURES.append(f"{path.name}: {len(old)} rows, dump has {len(rows)}")
        return

    worst, worst_at, bad = 0.0, "", 0
    for i, (a, b) in enumerate(zip(old, rows)):
        for k in a:
            x, y = a[k], str(b[k])
            if x == y:
                continue
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                bad += 1
                FAILURES.append(f"{path.name} row {i + 2} {k}: {x} against {y}")
                continue
            # An integer column is a count, a seed, a step or an outcome flag,
            # and none of those are allowed to move at all.
            if "." not in x and "e" not in x.lower():
                bad += 1
                FAILURES.append(f"{path.name} row {i + 2} {k}: {x} against {y}")
                continue
            rel = abs(fx - fy) / max(abs(fx), abs(fy), 1e-300)
            if rel > worst:
                worst, worst_at = rel, f"row {i + 2} {k}"
            if rel > RTOL:
                bad += 1
                FAILURES.append(f"{path.name} row {i + 2} {k}: {x} against {y}, "
                                f"relative {rel:.1e}")
    print(f"{path.name:22} {len(rows):6d} rows, worst relative difference "
          f"{worst:.1e} ({worst_at or 'none'}), {bad} over {RTOL:.0e}")


if __name__ == "__main__":
    export_physics()
    export_reward()
    export_oracle_layouts()
    if CHECK:
        if FAILURES:
            print(f"{len(FAILURES)} fixture cells disagree with a fresh dump:")
            for line in FAILURES[:20]:
                print(f"  {line}")
            print("regenerate with: python verify/export_golden.py")
            sys.exit(1)
        print("verify/golden is what src/rlarm/env.py produces today")
