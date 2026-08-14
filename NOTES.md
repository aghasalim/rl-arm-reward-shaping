# Decision trail

Written as I went. Every number here came from a run in this repo — nothing is
recalled from a paper or estimated. Where a result is noisy or a comparison is
unsound, it says so.

The success criterion was fixed **before any training** and never touched:

> End effector within **5 cm** of the target, **both joints below 0.1 rad/s**,
> held for **10 consecutive steps**, no obstacle contact, within **200 steps**.

The "hold" clause is what makes this a reward-design problem instead of a
reaching problem. An agent that *touches* the target is easy. One that arrives
and **stops** is not.

---

## Baseline, before anything was trained

A random policy scores **0% success** with a mean final distance of ~1.6 m. That
0% is what you want from a task: nothing is being handed to the agent for free.

---

## Exploit #1 — the agent kills itself to stop the bleeding

`v2` pays `-distance` every step and never penalises collision. Colliding
**terminates the episode**, and terminating stops the cost accruing.

The agent found this completely — **100% of evaluation episodes end in a
collision**, at around step 30, where a random policy survives past step 100. It
is not stumbling into the obstacle. It is steering into it, far earlier than
chance.

`v3` is the interesting failure, because it is what I actually wrote next. I
added a `-5` collision penalty and assumed that settled it. Nothing changed —
still 100% collision. The arithmetic I had never done:

```
survive to the step limit ≈ 200 × (−dist − 0.05) ≈ −100 or worse
crash immediately                                =   −5
```

A `-5` penalty against a `-100` alternative is not a deterrent, it is a discount.
**A termination penalty has to beat the worst case of staying alive.** Every
later version sets it by doing that subtraction explicitly.

---

## Exploit #2 — farming the drift in "provably safe" shaping

`v4` was meant to be the principled fix: potential-based shaping (Ng, Harada &
Russell 1999), `F = γΦ(s′) − Φ(s)` with `Φ = −distance`. There is a theorem
saying this cannot change the optimal policy. Collisions duly collapsed.

And the arm stopped doing anything — over 90% of episodes timed out, with a final
distance *worse than random*.

For a **stationary** agent, `Φ(s′) = Φ(s) = −d`, so

```
F = γ(−d) − (−d) = (1 − γ)·d       > 0
```

Positive income for standing still, and **larger the further away you stand**. At
γ=0.99 with my shaping weight of 5:

| distance held | per step | per 200-step episode |
|---|---|---|
| 1.0 m | +0.050 | +10.0 |
| 1.5 m | +0.075 | +15.0 |
| 2.0 m | +0.100 | **+20.0** |

The success bonus was `+20`. **Loitering at 2 m paid exactly as much as solving
the task, at zero collision risk.** An episode trace confirmed it exactly: 200
steps, return **+19.2**, the arm drifts from 1.80 m to 1.39 m and parks there.
It was not failing to learn. It learned precisely what I was paying for.

The theorem is not wrong — its policy invariance assumes an infinite horizon with
absorbing terminal states. Under a 200-step truncation that drift term stops
being telescoping bookkeeping and becomes income.

**Fix (v5):** γ=1 *in the shaping term only*, so it telescopes to
`w·(Φ_end − Φ_start)` over the episode and a stationary agent earns exactly zero
from it. The agent's own discount stays at 0.99.

---

## The bug that wasn't a reward bug

v5 removed both exploits and still solved nothing — 0% success, target reached in
6% of episodes. Two exploits down and the arm still would not do the task.

Instead of shaping further I wrote a **PD controller with exact inverse
kinematics** as an oracle — no learning, it already knows the answer — to ask
whether the criterion was achievable at all.

| `MAX_TORQUE` | oracle success (obstacle) | oracle success (no obstacle) |
|---|---|---|
| **2.0** (what I had) | 52.0% | **65.0%** |
| 4.0 | 66.0% | 83.5% |
| 6.0 | 70.0% | 90.0% |
| 8.0 | 73.5% | **97.0%** |
| 12.0 | 75.5% | 99.0% |

At my original torque limit, **a perfect controller that already knew the answer
failed a third of episodes with the obstacle removed.** The arm was
under-actuated: peak joint acceleration ~0.8 rad/s², so a large reorientation
could not be completed *and stopped* within 4 seconds. I had spent hours trying
to fix physically impossible episodes with reward functions.

Raising `MAX_TORQUE` to 8.0 took the oracle to 97%, and the learned policy from
2% to 12% on an unchanged reward.

**An oracle tells you whether the task is solvable. A reward curve never will.**

The oracle also fixes the honest reference point: 73.5% with obstacles, colliding
in 25.5% of episodes because a pure PD controller drives straight through them.
Beating it means avoiding obstacles it cannot — that, not the raw success number,
is what the learned policy has to justify.

---

## What didn't help: the velocity term

I assumed weighting joint speed in the potential would teach the arm to stop,
since stopping is the hard half. Swept at 2M steps, everything else fixed:

| `vel_weight` | success | reached target | settled given reached |
|---|---|---|---|
| 0.0 | 1.3% | **58.7%** | 2.3% |
| 0.1 | **12.0%** | 50.7% | 23.7% |
| 0.3 | 3.3% | 24.0% | 13.9% |
| 0.5 | 0.7% | 12.7% | 5.3% |

Backwards from my expectation, and monotonic above 0.1. More speed weighting does
make the arm settle more reliably *once it arrives* — but it arrives far less
often, because penalising speed everywhere discourages moving at all. The gain on
the last 5 cm is bought with a much larger loss on the first 1.5 m. It is a
milder relative of the v4 freeze. Kept at 0.1.

---

## The one that actually worked, and how I found it

At this point v5 had plateaued: **8.4% success at 8 million steps**, five seeds.
Going from 3M to 8M steps, adding a learning-rate decay, adding hold progress to
the observation and raising the time cost had between them changed nothing
outside noise. More compute was clearly not the answer.

So I stopped tuning and traced a single failing episode — one where the arm got
close and still failed:

```
 step   dist   maxvel  maxact
  178  0.1189   0.246   0.027
  185  0.0970   0.211   0.024
  192  0.0784   0.181   0.021
  199  0.0625   0.155   0.018     <- episode ends here, tolerance is 0.05
```

Two things jump out. The arm only arrives in the neighbourhood at **step 178 of
200**, and once there its action magnitude is **~0.02** — the motors are
essentially off. It is *coasting*, decelerating on damping alone, and the clock
runs out 1.2 cm short.

Then the aggregate over 17k steps of a trained policy:

```
velocity within tolerance:  24.6% of steps
distance within tolerance:   1.8% of steps
both:                        0.8% of steps
```

**Distance was the binding constraint, not velocity.** I had spent the entire
velocity-weight sweep tuning the wrong half of the criterion. The diagnostics
that made this visible (`reached_target_rate`, `settle_given_reached`) were
already in the evaluator; I had just been reading the success rate.

The cause is that a linear `Φ = −dist` pays the same per metre for the last 5 cm
as for the first, so there is almost no gradient left where precision matters.
Adding a `sqrt(dist)` term fixes exactly that — `d/dx √x → ∞` as `x → 0`, so the
final centimetres become worth several times more, and because it is still a
potential with γ=1 it telescopes and cannot be farmed.

| reward | steps | seeds | success |
|---|---|---|---|
| v5, linear `Φ = −dist` | 8M | 5 | **8.4% ± 4.9%** |
| v6, `Φ = −(dist + √dist)` | 3M | 3 | **44.7% ± 5.9%** |

A 5× improvement from one term, at less than half the compute. The lesson I
actually take from this project: **the shape of the potential near the goal
mattered far more than any amount of penalty tuning, and I only found it by
tracing one episode instead of staring at aggregate curves.**

---

## Final result

v6 at 3M steps, five seeds, scored on the fixed criterion over the same 200
held-out layouts:

| seed | success | collision | timeout | reached | settled given reached |
|---|---|---|---|---|---|
| 0 | 40.0% | 17.0% | 43.0% | 62.5% | 64.0% |
| 1 | 53.0% | 13.0% | 34.0% | 74.0% | 71.6% |
| 2 | 41.0% | 19.0% | 40.0% | 62.5% | 65.6% |
| 3 | 36.0% | 18.5% | 45.5% | 66.5% | 54.1% |
| 4 | 46.0% | 19.0% | 35.0% | 70.0% | 65.7% |
| **mean** | **43.2% ± 5.8%** | 17.3% | 39.5% | 67.1% | 64.2% |

Against a random policy at 0.0% and the PD oracle at 73.5%. The agent is well
short of the oracle on success and ahead of it on collisions (17.3% vs 25.5%),
which is the one thing a PD controller tracking an IK solution cannot do.

### Training longer made it substantially worse

Identical code, identical five seeds, only `--timesteps` changed:

| steps | success (5 seeds) |
|---|---|
| 3M | **43.2% ± 5.8%** |
| 8M | **18.0% ± 6.6%** |

This is the same phenomenon as the 12.0%-at-2M / 6.0%-at-4M result earlier, which
I had written off as single-seed noise. With five seeds either side it is not
noise. I verified `env.py` and `train.py` were untouched between the two runs
(both last modified 04:31, both runs launched after).

The one confound I cannot remove without more runs: the learning rate decays
linearly to zero *across the run*, so the 8M job is not the 3M job continued —
it sits at a high learning rate far longer before annealing. So the claim stays
narrow: **on this task, with a horizon-matched linear decay, 2.7× the compute
produced less than half the performance.** Whether that is late-training
collapse, the schedule, or an interaction between them is unresolved, and it is
the first thing I would investigate with more time.

---

## Where my methodology is weakest

Two things I would not defend if pushed:

**The sweeps are single-seed.** Every table above except the final result comes
from one seed per configuration. I have direct evidence that this is not enough:
the identical configuration scored **12.0% at 2M steps and 6.0% at 4M steps** —
more training, worse policy, same seed, same everything. That is either a genuine
late-training collapse or ordinary PPO variance, and with one seed I cannot tell
which. Differences between adjacent sweep rows should be read as suggestive, not
measured. I kept them because the *direction* of the `vel_weight` trend is
consistent across four values, and because hiding weak evidence would
misrepresent how the configuration was chosen.

**Reward is never compared across versions, because it cannot be.** v2 and v4 are
not in the same units and "v4 scores higher" would be a meaningless sentence.
Every table here is scored on the fixed external criterion, re-running each policy
on the same 200 held-out layouts (seeds 10000–10199, disjoint from training). The
training-curve plots show task success, never reward, for the same reason.

**A note on what changed between tables.** The v1–v4 comparison and the
`MAX_TORQUE` oracle sweep were run at different points in the project, and the
environment changed underneath them (torque limit, observation dimension). Rather
than quote numbers the current code cannot reproduce, every reward version was
retrained in the final environment for the results table. The earlier numbers are
kept in this narrative only where they describe the discovery itself, and are
labelled with the settings they were produced under.
