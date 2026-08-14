# Results

All policies evaluated on the same 200 held-out task layouts (seeds 10000+, disjoint from training), deterministic actions.

**Success criterion, fixed before any training:** end effector within 0.05 m of the target, both joints below 0.1 rad/s, held for 10 consecutive steps, no obstacle contact, within 200 steps.

## Reward versions

| reward | success | collision | timeout | final dist (m) | ep len | reached target | settled \| reached | speed in goal | mean collision step |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.0% | 45.0% | 55.0% | 1.652 | 150.7 | 10.0% | 0.0% | 2.16 | 90.4 |
| v1_sparse | 0.0% | 42.0% | 58.0% | 1.921 | 133.4 | 4.0% | 0.0% | 2.43 | 41.4 |
| v2_distance | 0.0% | 100.0% | 0.0% | 1.832 | 29.7 | 7.0% | 0.0% | 6.30 | 29.7 |
| v3_penalties | 0.0% | 100.0% | 0.0% | 1.691 | 26.3 | 4.0% | 0.0% | 5.39 | 26.3 |
| v4_potential | 0.0% | 4.0% | 96.0% | 1.512 | 193.7 | 9.0% | 0.0% | 0.52 | 41.9 |
| v5_progress | 0.0% | 11.5% | 88.5% | 0.594 | 183.4 | 22.0% | 0.0% | 0.11 | 55.6 |
| v6_goalfocus | 0.0% | 14.0% | 86.0% | 0.440 | 176.4 | 45.5% | 0.0% | 0.10 | 31.7 |

## Final policy, 5 seeds

| metric | mean | std | min | max |
|---|---|---|---|---|
| success_rate | 43.2% | 5.8% | 36.0% | 53.0% |
| collision_rate | 17.3% | 2.3% | 13.0% | 19.0% |
| timeout_rate | 39.5% | 4.4% | 34.0% | 45.5% |
| mean_final_dist | 0.371 | 0.045 | 0.291 | 0.432 |
| mean_len | 140.757 | 5.558 | 132.455 | 148.075 |

Per-seed success rate: seed 0: 40.0%, seed 1: 53.0%, seed 2: 41.0%, seed 3: 36.0%, seed 4: 46.0%

