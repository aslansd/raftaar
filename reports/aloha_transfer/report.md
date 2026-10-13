# Raftaar report — `aloha_transfer`

50 episodes · 20,000 frames · no training required


## No findings

No detector fired -- but this dataset was read with assumptions, so read that as *nothing was found* rather than *nothing is there*:

- visual sharding was not assessed


## Raw measurements


| phase | strategies | averaging hazard | H(a|s) |
|---|---|---|---|
| transport | 1 | 0.00x | 0.039 |
| release | 2 | 0.15x | 0.030 |

- Condition-space gap ratio: **0.0%**
- Intrinsic dimensionality of visited states: **3.31**
- Feature shards: **not assessed** — no observation.image_features in this dataset; visual sharding was not assessed

## How this dataset was read

- Source format: **lerobot** (codebase v3.0)
- Phase labels: **gripper** — derived, not present in the dataset
- Trajectories compared in: **first-3-joints**
