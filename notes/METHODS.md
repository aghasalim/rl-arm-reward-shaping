# Methods and full write-up

The long-form sections that used to live in the README, moved here unchanged.
The README summarises each one and links back to the heading. The working log
is in [NOTES.md](../NOTES.md).

---

## Abstract

A 2-link arm has to reach a target and stay there. Six reward functions were
written for it, and the agent exploited two of them in ways that were not
anticipated: the distance reward made it drive into the obstacle on purpose,
because a collision ends the episode and stops the accumulating distance penalty,
and the potential-based reward made it drift near the target and park, because
parking is worth more than the risk of overshooting.

Every policy is scored on the same fixed criterion over the same 200 held-out
layouts, disjoint from training, with deterministic actions. Training reward is
never compared across versions, because v2 and v4 are not in the same units.

The final agent succeeds on 43.2% ± 5.8% of episodes over five seeds. A
hand-written PD controller succeeds on 73.5%, which is stated rather than omitted.
The one axis on which the learned policy earns its existence is collisions: 17.3%
against the oracle's 25.5%, because a PD controller tracking an inverse-kinematics
solution drives straight through obstacles and the policy learned to go around.

Two results are reported that a tuning write-up would normally suppress. At 1.2M
steps the best reward still scores 0% success while reaching the target three
times more often than any earlier version, the reward change and the training
budget both had to be right, and either alone reads as failure. And training
longer made it worse, on identical code and seeds.

**Contributions.** (i) Two reward exploits documented with video, including the
mechanism. (ii) A fixed held-out evaluation criterion applied across all six
rewards. (iii) A non-learned oracle baseline that beats the agent, reported.
(iv) A negative result on training budget.

---

## 1. The task

A two-link torque-controlled arm (planar, viewed from above, so no gravity) has
to put its end effector on a randomly placed target and **hold it there**, while
a randomly placed obstacle sits somewhere in the workspace.

The success criterion was fixed **before any training** and never adjusted:

> Within **5 cm** of the target, **both joints under 0.1 rad/s**, held for
> **10 consecutive steps**, no obstacle contact, inside **200 steps**.

That "hold it there" clause is the whole difficulty. Training an arm to *touch*
a target is a tutorial exercise. Training one to arrive and **stop** is where
reward functions start getting exploited.

---

## 2. The two exploits

### 1. The agent killed itself to stop the bleeding

Reward v2 paid`-distance` every step, and never penalised hitting the obstacle.
Collision **ends the episode**: and ending the episode stops the cost accruing.

The agent found this in **200 out of 200** evaluation episodes. It collides at
step ~30, where a random policy survives past step 100. It isn't stumbling into
the obstacle; it's steering into it, far earlier than chance.

What makes this my favourite failure is v3, which is what I actually wrote next.
I added a`-5` collision penalty, assumed that was that, and it changed nothing
still 100% collision. The arithmetic I had never done:

```
survive to the step limit ≈ 200 × (−dist − 0.05) ≈ −100 or worse
crash immediately                                =   −5
```

A`-5` penalty against a`-100` alternative isn't a deterrent, it's a discount.
**A penalty has to beat the worst case of staying alive.**


### 2. The agent farmed my "provably safe" shaping

v4 was meant to be the principled fix: potential-based shaping from Ng, Harada &
Russell (1999),`F = γΦ(s′) − Φ(s)` with`Φ = −distance`. There's a theorem
saying this cannot change the optimal policy. Collisions duly dropped to ~7%.

And the arm stopped doing anything at all, 92.5% of episodes timed out, final
distance *worse than a random policy*.

For a **stationary** agent,`Φ(s′) = Φ(s) = −d`, so

```
F = γ(−d) − (−d) = (1 − γ)·d       > 0
```

A constant positive income for standing still, **larger the further away you
stand**. At γ=0.99 and my shaping weight:

| distance held | per step | per 200-step episode |
|---|---|---|
| 1.0 m | +0.050 | +10.0 |
| 1.5 m | +0.075 | +15.0 |
| 2.0 m | +0.100 | **+20.0** |

The success bonus was`+20`. **Loitering at 2 m paid exactly as much as solving
the task, at zero risk.** A single episode trace confirmed it: 200 steps, return
+19.2, the arm drifts to 1.39 m and parks.

The theorem isn't wrong. Its policy invariance assumes an infinite horizon with
absorbing terminal states; under a 200-step cutoff that drift term stops being
telescoping bookkeeping and becomes income. Setting **γ=1 in the shaping term
only** makes it telescope to`w·(Φ_end − Φ_start)` across the episode, so a
stationary agent earns exactly zero. The agent's own discount stays at 0.99.


---

## 3. The bug that wasn't a reward bug

After fixing both exploits the agent still solved nothing. Before shaping
anything further I wrote a **PD controller with exact inverse kinematics**: no
learning, it already knows the answer, to ask whether my criterion was
achievable at all.

|`MAX_TORQUE` | oracle success (obstacle) | oracle success (no obstacle) |
|---|---|---|
| **2.0** (what I had) | 52.0% | **65.0%** |
| 8.0 | 73.5% | **97.0%** |

At my original torque limit, **a perfect controller that already knew the answer
failed a third of episodes with the obstacle removed.** The arm was
under-actuated: peak joint acceleration ~0.8 rad/s², so a large reorientation
could not be completed *and stopped* within 4 seconds. I had been trying to fix
physically impossible episodes with reward functions.

**An oracle tells you whether the task is solvable. A reward curve never will.**

---

## 4. Results, in full

Rows 2 to 7 share a 1.2M-step budget so they are directly comparable; the final row
is the same reward given 3M steps and five seeds. Note that v6 at 1.2M still
scores **0%**: it reaches the target three times as often as any earlier version
but has not yet learned to stop. The reward change and the training budget both
had to be right.

Read the last two columns down the table: v1, v4 essentially never reach the
target, v5 reaches it 22% of the time and settles **never**, v6 at 3M reaches it
67% of the time and settles in 64% of those. The two exploits show up as the two
extremes, v2/v3 collide in *every* episode, v4 times out in almost every one.

**The hand-written PD controller still beats the agent on success, 73.5% to
43.2%, and I am not going to pretend otherwise.** What the agent does better is
the thing the oracle cannot do at all: it collides in 17.3% of episodes against
the oracle's 25.5%, because a PD controller tracking an IK solution drives
straight through obstacles. The policy learned to go around. That is the only
axis on which it earns its existence, and it is worth more than the headline
number.

### Training longer made it worse

2.7× the compute for less than half the performance. I had already seen a
smaller version of this (an identical config scoring 12.0% at 2M and 6.0% at 4M)
and assumed it was seed noise; with five seeds on each side it clearly is not.
One caveat I can't rule out: the learning rate decays linearly to zero *over the
run*, so the 8M run is not "the 3M run continued", it holds a high learning rate
for far longer. So the honest claim is narrow: **on this task, with a
horizon-matched decay, training longer was actively harmful.** I did not
diagnose it further, and it is the loose end I would pull on first.

---

## 5. Limitations

**The agent loses to a controller I wrote in an afternoon.** 43.2% against 73.5%.
It wins only on collisions (17.3% vs 25.5%). If the task were reaching alone,
there would be no case for RL here at all.

**Seed variance is large enough to swamp most comparisons.** The five final seeds
scored 36.0%, 40.0%, 41.0%, 46.0% and 53.0%. Any single-run improvement smaller
than about 15 points is indistinguishable from picking a lucky seed, which is
exactly why every sweep table in [NOTES.md](NOTES.md) is labelled as suggestive
rather than measured, those were run one seed at a time, before I understood how
wide the spread was.

**39.5% of episodes still time out.** The dominant remaining failure is not an
exploit and not a collision: the arm simply does not get there and stop within
200 steps.`settle_given_reached` of 64% says that once it arrives it usually
finishes;`reached_target_rate` of 67% says arriving at all is the bottleneck.

**More training made it worse and I don't know why.** See above. I can measure it
across five seeds; I cannot explain it.

**Everything here is one environment I designed myself.** A reward exploit I did
not build the conditions for is one I would not have found. The two I did find
are real, but a custom environment cannot tell you how a method generalises.


---

## 6. Docker image

Built and verified on`linux/arm64`: 1.98 GB, container reports healthy, and the
trained policy loads and evaluates from inside the image. The image carries only
the five models the showcase actually loads, the sweep arms, probe runs and the
8M comparison seeds are part of the write-up, not the app.

---

## 7. Method

**Environment**:`src/rlarm/env.py`. Standard rigid-body manipulator dynamics
(Spong ch. 7) with the closed-form 2-link mass matrix, integrated with RK4.

RK4 rather than Euler is not gold-plating. Under semi-implicit Euler at dt=0.02
the unforced arm *gains* energy, and an agent will happily learn to pump an
integrator artefact instead of solving the task, with nothing in the reward
curve to reveal it.`tests/test_physics.py` asserts energy conservation on the
unforced arm and fails under Euler.

**Observation (13)**: trig-encoded joint angles (so the policy never sees the
±π discontinuity), normalized velocities, end-effector position, vectors to
target and obstacle, and **hold progress**. That last one matters: termination
depends on how many consecutive steps the arm has been settled, and without it
in the observation the task isn't Markovian, the policy cannot tell whether one
more settled step ends the episode or whether the counter just reset.

**Algorithm**: PPO from Stable-Baselines3,`[128, 128]` MLP, 8 vectorized envs,
linear learning-rate decay.`DummyVecEnv` not`SubprocVecEnv`: an env step costs
~80 µs, below process-IPC overhead, so subprocess workers measure *slower*. No
`VecNormalize`, normalizing reward would rescale each reward version
differently, which is exactly the confound this project exists to avoid.

---

## 8. What I'd do next

In rough order of how much I think each would pay off:

1. **Explain the 3M-vs-8M collapse.** Training 2.7× longer more than halved
   success across five seeds and I can't account for it. The obvious first
   experiment is a constant learning rate at both horizons, which separates
   "longer training hurts" from "the linear decay schedule hurts when stretched".
2. **Attack the timeout, not the reward.** 39.5% of episodes end with the arm
   still travelling.`reached_target_rate` says arriving is the bottleneck, not
   settling, so the next gain is in getting there faster, not in more shaping.
   The reward is no longer the limiting factor and I should stop tuning it.
3. **A second algorithm.** SAC is off-policy and far more sample-efficient on
   continuous control; on a task where more PPO steps actively hurt, that is the
   natural comparison rather than a box-ticking exercise.
4. **Curriculum on the obstacle.** The collision rate sits at 17% and barely
   moves. Starting without the obstacle and phasing it in would test whether the
   policy is genuinely avoiding it or has just learned a cautious posture.

What I would *not* do is tune the reward further. The two exploits are fixed, the
potential's shape near the goal is fixed, and everything since has been noise.

## Additional detail

Paragraphs kept from the earlier, longer README.

**[▶ Live demo](https://rl-arm-reward-shaping.streamlit.app/)**: the two exploits
on video, the reward-shaping timeline, and the per-seed spread.

A reinforcement learning project by a third-year Applied Computer Science (AI)
student. I built the environment from scratch, wrote six reward functions, and
the agent cheated two of them in ways I did not see coming.

**This README leads with the things that went wrong, because those are the part
worth reading.** The final agent is not very good, I say exactly how not-good
below, and compare it against a hand-written controller that beats it. What I
can defend is every number here, how it was measured, and why the reward
function ended up the shape it did.

The full working log is in **[NOTES.md](NOTES.md)**.

*v2. The arm makes straight for the red circle. Episodes are over in ~30 steps.*

*v4. It drifts a little, then stops and waits out the clock. It is being paid to
do this.*

### The final agent, at three points in training

Same three task layouts in every clip, so this is like-for-like, a later
checkpoint cannot look better by drawing easier targets.

Same code, same five seeds, only`--timesteps` changed:

```bash
make setup && make test
```

Check the task is solvable before training anything (this is the habit the
project is really about):

Reproduce the reward-shaping story, then the final policy:

```bash
make shaping && make final && make eval && make plots && make videos
```

`make final` is 8M steps × 5 seeds and takes roughly 40 minutes on 10 CPU cores.
Everything runs on CPU; there is no GPU path and it wouldn't help at this size.
