"""ReachAvoid-v0, a planar two-link arm that must reach a target and *stop* there
without hitting an obstacle.

Why a custom environment
------------------------
The reward pitfalls of LunarLander and BipedalWalker are documented in a hundred
blog posts, so "shaping" them is really just recalling someone else's answer.
Designing the observation space, the termination rules and the reward from
scratch is where the actual failure modes show up, and they are mine.

Physics
-------
Planar SCARA-style arm viewed from above, so there is no gravity term. The
dynamics are the standard rigid-body manipulator equations (Spong, *Robot
Modeling and Control*, ch. 7):

    M(q) q̈ + C(q, q̇) q̇ + b q̇ = τ

with the closed-form 2-link mass matrix. Links are modelled as thin uniform rods,
so I = m·l²/12 about the centre of mass and lc = l/2.

Integrated with RK4 rather than Euler. That is not gold-plating: with semi-implicit
Euler at dt=0.02 the unforced arm gains energy steadily, which would mean the
agent is learning to exploit an integrator artefact rather than the task. The
energy-conservation test in tests/test_physics.py is what checks this, and it
fails under Euler.

The task
--------
Reach a randomly placed target with the end effector and hold it there. "Hold" is
the part that makes reward design interesting: an agent that merely *touches* the
target is easy to train, and useless for anything resembling a real manipulator.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- geometry / physics constants -----------------------------------------
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
LC1, LC2 = L1 / 2, L2 / 2
I1, I2 = M1 * L1**2 / 12, M2 * L2**2 / 12
DAMPING = 0.10
# MAX_TORQUE was 2.0 for the v1-v4 experiments, which made the arm
# under-actuated: peak joint acceleration ~0.8 rad/s², so a large reorientation
# could not be completed *and brought to a stop* inside the 200-step limit. A PD
# oracle with exact inverse kinematics topped out at 65% success with the
# obstacle disabled, i.e. a third of episodes were physically unwinnable and no
# reward function could have fixed them. At 8.0 the same oracle reaches 97%.
# See NOTES.md, "the bug that was not a reward bug".
MAX_TORQUE = 8.0
MAX_VEL = 12.0
DT = 0.02
SUBSTEPS = 2

# --- task constants --------------------------------------------------------
MAX_STEPS = 200
GOAL_TOL = 0.05          # metres: end effector within 5 cm of target
VEL_TOL = 0.10           # rad/s: both joints essentially stopped
HOLD_STEPS = 10          # consecutive steps inside tolerance before success
OBSTACLE_R = 0.25
REWARD_VERSIONS = ("v1_sparse", "v2_distance", "v3_penalties", "v4_potential",
                   "v5_progress", "v6_goalfocus")


def _mass_matrix(q2: float) -> np.ndarray:
    c2 = math.cos(q2)
    m11 = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2)
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * c2)
    m22 = I2 + M2 * LC2**2
    return np.array([[m11, m12], [m12, m22]], dtype=np.float64)


def _coriolis(q2: float, dq1: float, dq2: float) -> np.ndarray:
    """C(q, q̇)·q̇ for the 2-link planar arm."""
    h = M2 * L1 * LC2 * math.sin(q2)
    return np.array([-h * dq2**2 - 2 * h * dq1 * dq2, h * dq1**2], dtype=np.float64)


def _deriv(state: np.ndarray, tau: np.ndarray) -> np.ndarray:
    q1, q2, dq1, dq2 = state
    M = _mass_matrix(q2)
    c = _coriolis(q2, dq1, dq2)
    ddq = np.linalg.solve(M, tau - c - DAMPING * np.array([dq1, dq2]))
    return np.array([dq1, dq2, ddq[0], ddq[1]], dtype=np.float64)


def kinetic_energy(state: np.ndarray) -> float:
    """T = ½ q̇ᵀ M(q) q̇. With no gravity and no damping this is the total energy,
    which is what the physics test asserts is conserved."""
    dq = state[2:]
    return float(0.5 * dq @ _mass_matrix(state[1]) @ dq)


def forward_kinematics(q1: float, q2: float) -> tuple[np.ndarray, np.ndarray]:
    elbow = np.array([L1 * math.cos(q1), L1 * math.sin(q1)])
    ee = elbow + np.array([L2 * math.cos(q1 + q2), L2 * math.sin(q1 + q2)])
    return elbow, ee


def _seg_point_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Shortest distance from point p to segment ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class ReachAvoidEnv(gym.Env):
    """Two-link arm, randomized target and obstacle.

    Observation (13): cos/sin of both joints, normalized joint velocities,
    end-effector position, vector to target, vector to obstacle centre, and
    hold progress. Trig encoding rather than raw angles so the policy never
    sees the discontinuity at ±π.

    Action (2): joint torques in [-1, 1], scaled by MAX_TORQUE.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, reward_version: str = "v6_goalfocus", render_mode=None,
                 shaping_weight: float = 10.0, collision_penalty: float = 20.0,
                 vel_weight: float = 0.1, use_obstacle: bool = True,
                 max_torque: float = MAX_TORQUE, time_cost: float = 0.03,
                 goal_focus: float = 1.0):
        """shaping_weight / collision_penalty / use_obstacle are exposed because
        they are what the v5 balance sweep varies (see NOTES.md). use_obstacle=False
        is the diagnostic that separates "cannot reach" from "will not risk the
        obstacle", without it those two failures look identical from outside."""
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(f"reward_version must be one of {REWARD_VERSIONS}")
        self.reward_version = reward_version
        self.render_mode = render_mode
        self.shaping_weight = shaping_weight
        self.collision_penalty = collision_penalty
        self.vel_weight = vel_weight
        self.use_obstacle = use_obstacle
        # Per-instance so the under-actuation experiment is reproducible from
        # code rather than by editing a module constant.
        self.max_torque = max_torque
        # Raised from 0.01 after measuring 73% timeouts: at 0.01 the whole
        # episode's time cost (-2) was noise against ~+15 of shaping, so there
        # was no pressure to actually finish. The collision penalty has to stay
        # well above the total time cost or exploit #1 returns: at 0.03 the
        # worst case for surviving is -6, against a -20 collision.
        self.time_cost = time_cost
        # Weight on a sqrt(dist) term in the potential. A purely linear -dist
        # potential pays the same per metre for the last 5 cm as for the first,
        # and the measured failure was precision, not settling: across 17k steps
        # of a trained policy, velocity was inside tolerance 24.6% of the time
        # but distance only 1.8%. d/dx sqrt(x) blows up as x -> 0, so this makes
        # the final centimetres worth roughly 3x more. Still potential-based, so
        # it cannot be farmed.
        self.goal_focus = goal_focus
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        # 13th element is hold progress. Without it this task is not Markovian:
        # termination depends on _hold, the count of consecutive settled steps,
        # and a policy that cannot observe _hold cannot know whether one more
        # settled step ends the episode or whether it just reset the counter.
        high = np.array([1, 1, 1, 1, 1, 1, 2.1, 2.1, 4.2, 4.2, 4.2, 4.2, 1],
                        dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.state = np.zeros(4)
        self.target = np.zeros(2)
        self.obstacle = np.zeros(2)
        self._hold = 0
        self._steps = 0
        self._prev_dist = 0.0
        self._prev_pot = 0.0

    #: setup -------------------------------------------------------------
    def _sample_task(self) -> None:
        """Place obstacle and target so the episode is solvable.

        The target is rejected if it sits inside the obstacle or is unreachable.
        Without the reachability check ~6% of episodes are impossible, which
        silently caps the success rate and looks like a training failure.
        """
        rng = self.np_random
        for _ in range(200):
            ang = rng.uniform(-math.pi, math.pi)
            rad = rng.uniform(0.6, 1.2)
            self.obstacle = np.array([rad * math.cos(ang), rad * math.sin(ang)])
            t_ang = rng.uniform(-math.pi, math.pi)
            t_rad = rng.uniform(0.4, L1 + L2 - 0.15)
            self.target = np.array([t_rad * math.cos(t_ang), t_rad * math.sin(t_ang)])
            if np.linalg.norm(self.target - self.obstacle) > OBSTACLE_R + 0.15:
                return
        # Fall back to a guaranteed-valid layout rather than looping forever.
        self.obstacle = np.array([0.0, 1.0])
        self.target = np.array([1.0, -0.5])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_task()
        # Start near the "elbow out" home pose with a small random perturbation,
        # and never in collision.
        for _ in range(100):
            self.state = np.array(
                [self.np_random.uniform(-math.pi, math.pi),
                 self.np_random.uniform(-2.5, 2.5), 0.0, 0.0]
            )
            if not self._in_collision():
                break
        self._hold = 0
        self._steps = 0
        self._prev_dist = self._dist()
        # Joint velocities are zero at reset, so the speed term contributes 0.
        self._prev_pot = self._potential(self._prev_dist, 0.0)
        return self._obs(), {}

    #: helpers -----------------------------------------------------------
    def _ee(self) -> np.ndarray:
        return forward_kinematics(self.state[0], self.state[1])[1]

    def _dist(self) -> float:
        return float(np.linalg.norm(self._ee() - self.target))

    def _in_collision(self) -> bool:
        if not self.use_obstacle:
            return False
        elbow, ee = forward_kinematics(self.state[0], self.state[1])
        base = np.zeros(2)
        return (
            _seg_point_dist(base, elbow, self.obstacle) < OBSTACLE_R
            or _seg_point_dist(elbow, ee, self.obstacle) < OBSTACLE_R
        )

    def _potential(self, dist: float, speed: float) -> float:
        """Φ for the telescoping (γ=1) shaping used by v5 and v6.

        v5 is linear in distance. v6 adds goal_focus·sqrt(dist), whose gradient
        grows without bound as dist -> 0, so the final centimetres are worth
        several times more than the same distance travelled far away. That single
        term moved success from 8.4% to 44.7%.
        """
        gf = self.goal_focus if self.reward_version == "v6_goalfocus" else 0.0
        return -(dist + gf * math.sqrt(dist) + self.vel_weight * speed)

    def _settled(self) -> bool:
        return self._dist() < GOAL_TOL and np.all(np.abs(self.state[2:]) < VEL_TOL)

    def _obs(self) -> np.ndarray:
        q1, q2, dq1, dq2 = self.state
        ee = self._ee()
        return np.array(
            [math.cos(q1), math.sin(q1), math.cos(q2), math.sin(q2),
             dq1 / MAX_VEL, dq2 / MAX_VEL, ee[0], ee[1],
             self.target[0] - ee[0], self.target[1] - ee[1],
             self.obstacle[0] - ee[0], self.obstacle[1] - ee[1],
             self._hold / HOLD_STEPS],
            dtype=np.float32,
        )

    #: dynamics ----------------------------------------------------------
    def _integrate(self, tau: np.ndarray) -> None:
        h = DT / SUBSTEPS
        s = self.state.astype(np.float64)
        for _ in range(SUBSTEPS):
            k1 = _deriv(s, tau)
            k2 = _deriv(s + 0.5 * h * k1, tau)
            k3 = _deriv(s + 0.5 * h * k2, tau)
            k4 = _deriv(s + h * k3, tau)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s[2:] = np.clip(s[2:], -MAX_VEL, MAX_VEL)
        s[0] = (s[0] + math.pi) % (2 * math.pi) - math.pi
        s[1] = (s[1] + math.pi) % (2 * math.pi) - math.pi
        self.state = s

    def step(self, action):
        tau = np.clip(np.asarray(action, dtype=np.float64), -1, 1) * self.max_torque
        self._integrate(tau)
        self._steps += 1

        dist = self._dist()
        collided = self._in_collision()
        self._hold = self._hold + 1 if self._settled() else 0
        success = self._hold >= HOLD_STEPS

        reward = self._reward(dist, tau, collided, success)
        self._prev_dist = dist

        terminated = bool(success or collided)
        truncated = bool(self._steps >= MAX_STEPS and not terminated)
        info = {
            "success": success, "collision": collided, "dist": dist,
            "steps": self._steps, "is_success": success,
        }
        return self._obs(), float(reward), terminated, truncated, info

    #: reward ------------------------------------------------------------
    def _reward(self, dist, tau, collided, success) -> float:
        """The four versions are kept side by side on purpose: NOTES.md refers to
        them by name, and being able to re-run any earlier one is what makes the
        shaping story reproducible instead of anecdotal."""
        v = self.reward_version

        if v == "v1_sparse":
            return 1.0 if success else 0.0

        if v == "v2_distance":
            return (10.0 if success else 0.0) - dist

        if v == "v3_penalties":
            # The naive next step: nudge it to hurry and to avoid the obstacle.
            r = -dist - 0.05
            if collided:
                r -= 5.0
            if success:
                r += 10.0
            return r

        if v == "v4_potential":
            # Potential-based shaping (Ng, Harada & Russell 1999), Φ = -dist.
            # This version is KEPT BROKEN on purpose: it is exploit #2 in
            # NOTES.md. F = γΦ(s') - Φ(s) pays a *stationary* agent
            # (1-γ)·dist every step: +20 per episode at 2 m, exactly the success
            # bonus, for standing still at zero collision risk. The policy
            # invariance theorem assumes an infinite horizon with absorbing
            # terminal states; under a 200-step truncation the drift is income.
            gamma = 0.99
            shaping = 5.0 * (gamma * (-dist) - (-self._prev_dist))
            r = shaping - 0.002 * float(tau @ tau)
            if dist < GOAL_TOL:
                r += 0.5
            if collided:
                r -= 30.0
            if success:
                r += 20.0
            return r

        # v5: γ=1, so the shaping term telescopes to 5·(Φ_end - Φ_start) over the
        # episode and a stationary agent earns exactly nothing from it. The
        # potential also includes joint speed, which makes decelerating on
        # approach rewarded directly instead of only via the terminal bonus --
        # the task requires stopping, not just arriving.
        # Because γ=1 the whole term telescopes to w·(d₀ - d_end - k·v_end): the
        # speed component can only be earned by *finishing* slow, never by
        # loitering slowly. That is what lets vel_weight be raised aggressively
        # without recreating the v4 freeze.
        pot = self._potential(dist, float(np.linalg.norm(self.state[2:])))
        shaping = self.shaping_weight * (pot - self._prev_pot)
        self._prev_pot = pot
        # Normalised by MAX_TORQUE² so the effort term keeps the same weight if
        # the actuator limit changes: otherwise raising torque silently turns a
        # rounding-error penalty into the dominant term.
        effort = 0.002 * float(tau @ tau) / (self.max_torque ** 2)
        r = shaping - effort - self.time_cost
        # Pays only while *actually* satisfying the success condition. It cannot
        # be farmed: 10 consecutive settled steps terminate the episode, so this
        # term is capped at +10 by construction, unlike the v4 in-goal bonus
        # which paid for merely being nearby at any speed. Added because the
        # measured settling speed was 0.11-0.12 rad/s against a 0.10 threshold --
        # the policy was stopping just barely too late to score.
        if self._settled():
            r += 1.0
        if collided:
            # Worst case for simply surviving is -2 (200 x -0.01), so ending the
            # episode early can never be the cheap way out. This is the fix for
            # exploit #1.
            r -= self.collision_penalty
        if success:
            r += 50.0
        return r

    #: rendering ---------------------------------------------------------
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = plt.Figure(figsize=(4, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0, 0, 1, 1)  # the arm, not the margins
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.axis("off")

        elbow, ee = forward_kinematics(self.state[0], self.state[1])
        ax.add_patch(plt.Circle(tuple(self.obstacle), OBSTACLE_R, color="#d62728", alpha=0.35))
        ax.add_patch(plt.Circle(tuple(self.target), GOAL_TOL, color="#2ca02c"))
        ax.add_patch(plt.Circle(tuple(self.target), 0.15, color="#2ca02c", alpha=0.15))
        ax.plot([0, elbow[0]], [0, elbow[1]], "-", lw=5, color="#1f77b4")
        ax.plot([elbow[0], ee[0]], [elbow[1], ee[1]], "-", lw=5, color="#4c9ed9")
        ax.plot([0], [0], "o", ms=9, color="#333")
        ax.plot([ee[0]], [ee[1]], "o", ms=7, color="#ff7f0e")
        ax.text(-2.1, 2.0, f"{self.reward_version}  step {self._steps}  d={self._dist():.3f}",
                fontsize=8, color="#333")

        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, :3].copy()


gym.register(id="ReachAvoid-v0", entry_point=f"{__name__}:ReachAvoidEnv")
