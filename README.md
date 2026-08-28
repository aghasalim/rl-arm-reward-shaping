# Teaching an arm to stop, not just arrive

[![ci](https://github.com/aghasalim/rl-arm-reward-shaping/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/rl-arm-reward-shaping/actions/workflows/ci.yml)
[![demo-link](https://github.com/aghasalim/rl-arm-reward-shaping/actions/workflows/demo.yml/badge.svg)](https://github.com/aghasalim/rl-arm-reward-shaping/actions/workflows/demo.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**[▶ Live demo](https://rl-arm-reward-shaping.streamlit.app/)**: the exploits on
video and the per-seed spread.

A 2-link torque-controlled arm has to reach a randomly placed target and stay
there, with an obstacle in the workspace. I built the environment, wrote six
reward functions, and the agent exploited two of them in ways I did not see
coming. The final agent scores **43.2% ± 5.8%** success over five seeds, against
**73.5%** for a hand-written PD controller, which I am not going to hide. It wins
on one axis, collisions, 17.3% against the oracle's 25.5%, because a PD
controller tracking an inverse-kinematics solution drives straight through
obstacles and the policy learned to go around.

Full write-up in **[notes/METHODS.md](notes/METHODS.md)**, working log in
**[NOTES.md](NOTES.md)**.

## The task

The success criterion was fixed **before any training** and never adjusted:

> Within **5 cm** of the target, **both joints under 0.1 rad/s**, held for
> **10 consecutive steps**, no obstacle contact, inside **200 steps**.

Touching a target is a tutorial exercise; arriving and stopping is where reward
functions get exploited. [More](notes/METHODS.md#1-the-task).

## The two exploits

**v2, distance reward.** It paid `-distance` every step with no collision penalty,
and a collision ends the episode, which stops the cost accruing. The agent found
this in **200 out of 200** evaluation episodes, and the `-5` penalty I added in v3
was no deterrent against a `-100` alternative.
[Arithmetic](notes/METHODS.md#1-the-agent-killed-itself-to-stop-the-bleeding).

![v2 drives into the obstacle on purpose](reports/figures/v2_suicide.gif)

*v2. The arm makes straight for the red circle, in ~30 steps.*

**v4, potential-based shaping**, from Ng, Harada & Russell (1999): `F = γΦ(s′) −
Φ(s)` with `Φ = −distance`. Collisions dropped to ~7% and the arm stopped moving,
92.5% of episodes timed out, because a stationary agent earns `(1 − γ)·d` per step
and loitering at 2 m pays `+20.0` an episode, exactly the success bonus. Policy
invariance assumes an infinite horizon; under a 200-step cutoff that term is
income. [Derivation](notes/METHODS.md#2-the-agent-farmed-my-provably-safe-shaping).

![v4 drifts and then parks](reports/figures/v4_freeze.gif)

*v4. It drifts, then parks and waits out the clock, and is paid to.*

## The bug that wasn't a reward bug

Both exploits fixed, the agent still solved nothing, so I wrote a PD controller
with exact inverse kinematics to ask whether the criterion was achievable at all.
At my original torque limit it scored **65.0%** with the obstacle removed: the arm
was under-actuated, and I had been trying to fix impossible episodes with reward
functions. [Torque table](notes/METHODS.md#3-the-bug-that-wasnt-a-reward-bug).

## Results

Same three layouts in every clip, so a later checkpoint cannot look better by
drawing easier targets.

| 200k steps | 2M steps | 3M steps (final) |
|---|---|---|
| ![early](reports/figures/final_early.gif) | ![mid](reports/figures/final_mid.gif) | ![late](reports/figures/final_late.gif) |
| 3 timeouts | 2 successes, 1 timeout | 3 successes |

![success rate across reward versions](reports/figures/reward_shaping_comparison.png)

![per-seed spread](reports/figures/multiseed.png)

Every policy is scored on the same fixed criterion over the same 200 held-out
layouts (seeds 10000 to 10199, disjoint from training), deterministic actions.
Training reward is never compared across versions. Full table in
[reports/results.md](reports/results.md).

| reward | steps | success | collision | timeout | reached target | settled given reached | final dist |
|---|---|---|---|---|---|---|---|
| random policy | - | 0.0% | 45.0% | 55.0% | 10.0% | 0.0% | 1.652 m |
| v1 sparse | 1.2M | 0.0% | 42.0% | 58.0% | 4.0% | 0.0% | 1.921 m |
| v2 distance | 1.2M | 0.0% | **100.0%** | 0.0% | 7.0% | 0.0% | 1.832 m |
| v3 penalties | 1.2M | 0.0% | **100.0%** | 0.0% | 4.0% | 0.0% | 1.691 m |
| v4 potential | 1.2M | 0.0% | 4.0% | **96.0%** | 9.0% | 0.0% | 1.512 m |
| v5 progress | 1.2M | 0.0% | 11.5% | 88.5% | 22.0% | 0.0% | 0.594 m |
| v6 goal-focus | 1.2M | 0.0% | 14.0% | 86.0% | **45.5%** | 0.0% | 0.440 m |
| **v6 goal-focus** (5 seeds) | 3M | **43.2% ± 5.8%** | 17.3% | 39.5% | 67.1% | 64.2% | **0.371 m** |
| *PD oracle (no learning)* | - | *73.5%* | *25.5%* | - | - | - | - |

Rows 2 to 7 share a 1.2M-step budget. v6 at 1.2M still scores 0% while reaching
the target three times as often as any earlier version: the reward change and the
budget both had to be right.
[Column by column](notes/METHODS.md#4-results-in-full).

### Training longer made it worse

Same code, same five seeds, only `--timesteps` changed:

| training steps | success (5 seeds) |
|---|---|
| 3M | **43.2% ± 5.8%** |
| 8M | **18.0% ± 6.6%** |

![the 8M run peaks and then slides back](reports/figures/longer_training.gif)

*Both runs as they train, five seeds each. The 8M run reaches roughly where
the 3M run finishes and then slides back down to 18.0%, which is the part a
final number alone would hide.*

2.7× the compute for less than half the performance. The learning rate decays
linearly over the run, so the 8M run is not "the 3M run continued".
[Caveat](notes/METHODS.md#training-longer-made-it-worse).

## Limitations

The agent loses to a controller I wrote in an afternoon, 43.2% against 73.5%, and
wins only on collisions. Seed variance is wide: the five final seeds scored 36.0%,
40.0%, 41.0%, 46.0% and 53.0%, so anything smaller than about 15 points is
indistinguishable from a lucky seed. 39.5% of episodes still time out, and this is
one environment I designed myself. [All five](notes/METHODS.md#5-limitations).

## Running it

```bash
make setup && make test
make oracle
make shaping && make final && make eval && make plots && make videos
make showcase
```

`make oracle` checks the task is solvable before training anything, the habit the
project is really about. `make final` is 8M steps × 5 seeds, roughly 40 minutes on
10 CPU cores, CPU only.

```bash
docker build -t rl-arm-reward-shaping . && docker run -p 8501:8501 rl-arm-reward-shaping
```

[What is in the image](notes/METHODS.md#6-docker-image).

## Method

Rigid-body manipulator dynamics (Spong ch. 7) integrated with RK4, not Euler:
under semi-implicit Euler the unforced arm gains energy and an agent will learn to
pump the artefact instead of solving the task. PPO from Stable-Baselines3,
`[128, 128]` MLP, 13-dimensional observation including hold progress, no
`VecNormalize`. [Full method](notes/METHODS.md#7-method).

```
src/rlarm/
  env.py        custom Gymnasium env, all six reward versions side by side
  oracle.py     PD + inverse-kinematics feasibility oracle
  train.py      PPO training with per-episode logging and checkpoints
  evaluate.py   fixed success criterion + reward-hacking diagnostics
  sweep.py      parallel configuration sweeps
  report.py     scores every policy, writes reports/results.md
  plots.py      training curves (task metrics, never reward)
  record.py     GIF recording on fixed seeds
app/showcase.py Streamlit showcase
tests/          physics and env-contract tests
```

## What I'd do next

The 3M vs 8M collapse is the loose end I would pull on first. I would not tune the
reward further. [Four ideas, ranked](notes/METHODS.md#8-what-id-do-next).

## References

The papers and sources this implementation follows. Each one is here because
the code uses the method, the dataset or the metric it describes.

- **Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. 2017.** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) the algorithm used.
- **Ng, Harada, Russell. Policy Invariance Under Reward Transformations. ICML 1999.** potential based shaping, and the condition under which shaping does not change the optimal policy.
- **Raffin, Hill, Gleave et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22, 2021.** the PPO implementation.

## Author and licence

Aghasalim Mustafazada. MIT, see [LICENSE](LICENSE).
