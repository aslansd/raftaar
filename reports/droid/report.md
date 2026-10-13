# Raftaar report — `droid`

50 episodes · 14,357 frames · no training required


## 1 findings (0 critical)


### [INFO] IDLE_CHANNEL — gripper in 'approach'

**Observed.** no variance in this phase

**Consequence.** Usually by design -- a gripper is idle during an approach. Re-run with `--reference` against a dataset you trust to tell design from defect.


## Raw measurements


| phase | strategies | averaging hazard | H(a|s) |
|---|---|---|---|
| transport | 2 | 0.12x | 0.041 |
| release | 3 | 0.20x | 0.090 |
| approach | 3 | 0.53x | 0.156 |

- Condition-space gap ratio: **0.0%**
- Intrinsic dimensionality of visited states: **5.91**
- Feature shards: **not assessed** — no observation.image_features in this dataset; visual sharding was not assessed

## How this dataset was read

- Source format: **lerobot** (codebase v3.0)
- Phase labels: **gripper** — derived, not present in the dataset
- Trajectories compared in: **first-3-joints**
