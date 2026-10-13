# Raftaar report — `aloha_insertion`

50 episodes · 25,000 frames · no training required


## No findings

No detector fired -- but this dataset was read with assumptions, so read that as *nothing was found* rather than *nothing is there*:

- visual sharding was not assessed


## Raw measurements


| phase | strategies | averaging hazard | H(a|s) |
|---|---|---|---|
| transport | 2 | 0.28x | 0.033 |
| release | 1 | 0.00x | 0.034 |

- Condition-space gap ratio: **0.0%**
- Intrinsic dimensionality of visited states: **4.58**
- Feature shards: **not assessed** — no observation.image_features in this dataset; visual sharding was not assessed

## How this dataset was read

- Source format: **lerobot** (codebase v3.0)
- Phase labels: **gripper** — derived, not present in the dataset
- Trajectories compared in: **first-3-joints**
