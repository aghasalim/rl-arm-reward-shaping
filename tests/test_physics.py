"""Physics and environment-contract checks.

The energy test is the important one. An integrator that quietly injects energy
turns the task into a different problem -- the agent learns to pump the
integrator instead of controlling the arm, and nothing about the reward curve
would reveal it. This test fails under semi-implicit Euler at dt=0.02, which is
why the environment uses RK4.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.rlarm import env as E


def test_unforced_arm_conserves_energy():
    """No gravity, no damping, no torque -> total energy is constant."""
    old_damping = E.DAMPING
    E.DAMPING = 0.0
    try:
        e = E.ReachAvoidEnv()
        e.reset(seed=0)
        e.state = np.array([0.3, -0.8, 2.0, -1.5])
        e0 = E.kinetic_energy(e.state)
        for _ in range(500):
            e._integrate(np.zeros(2))
        drift = abs(E.kinetic_energy(e.state) - e0) / e0
        assert drift < 1e-3, f"energy drifted {drift:.2%} -- integrator is injecting energy"
    finally:
        E.DAMPING = old_damping


def test_damping_removes_energy_monotonically():
    e = E.ReachAvoidEnv()
    e.reset(seed=0)
    e.state = np.array([0.3, -0.8, 3.0, -2.0])
    energies = []
    # 5000 steps = 100 s. The effective inertia of this arm is ~2.4, not 1, so
    # with b=0.1 the energy time-constant is long: measured decay is 4.3% of the
    # initial energy still remaining at 2000 steps and 0.05% at 5000. The horizon
    # is set from that measurement rather than from a back-of-envelope guess.
    for _ in range(5000):
        e._integrate(np.zeros(2))
        energies.append(E.kinetic_energy(e.state))
    assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))
    assert energies[-1] < 0.01 * energies[0]


def test_forward_kinematics_known_poses():
    _, ee = E.forward_kinematics(0.0, 0.0)
    assert np.allclose(ee, [E.L1 + E.L2, 0.0], atol=1e-9)
    _, ee = E.forward_kinematics(math.pi / 2, 0.0)
    assert np.allclose(ee, [0.0, E.L1 + E.L2], atol=1e-9)
    # Folded back on itself: the end effector returns to the base.
    _, ee = E.forward_kinematics(0.0, math.pi)
    assert np.allclose(ee, [0.0, 0.0], atol=1e-9)


def test_segment_distance():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert E._seg_point_dist(a, b, np.array([0.5, 0.5])) == pytest.approx(0.5)
    # Beyond the segment end, distance is to the endpoint, not the infinite line.
    assert E._seg_point_dist(a, b, np.array([2.0, 0.0])) == pytest.approx(1.0)


@pytest.mark.parametrize("version", E.REWARD_VERSIONS)
def test_env_contract(version):
    e = E.ReachAvoidEnv(reward_version=version)
    obs, _ = e.reset(seed=1)
    assert e.observation_space.contains(obs), "reset obs outside declared space"
    for _ in range(50):
        obs, r, term, trunc, info = e.step(e.action_space.sample())
        assert e.observation_space.contains(obs), "step obs outside declared space"
        assert math.isfinite(r)
        if term or trunc:
            break


def test_target_is_always_reachable_and_clear_of_obstacle():
    e = E.ReachAvoidEnv()
    for seed in range(200):
        e.reset(seed=seed)
        assert np.linalg.norm(e.target) <= E.L1 + E.L2, "target outside workspace"
        assert np.linalg.norm(e.target - e.obstacle) > E.OBSTACLE_R, "target inside obstacle"


def test_episode_terminates_within_step_limit():
    e = E.ReachAvoidEnv(reward_version="v1_sparse")
    e.reset(seed=3)
    for i in range(E.MAX_STEPS + 5):
        _, _, term, trunc, _ = e.step(np.zeros(2, dtype=np.float32))
        if term or trunc:
            assert i + 1 <= E.MAX_STEPS
            return
    pytest.fail("episode never ended")
