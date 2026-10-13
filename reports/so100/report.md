# Raftaar report — `so100`

50 episodes · 19,631 frames · no training required


## No findings

No detector fired -- but this dataset was read with assumptions, so read that as *nothing was found* rather than *nothing is there*:

- visual sharding was not assessed


## Raw measurements


| phase | strategies | averaging hazard | H(a|s) |
|---|---|---|---|
| transport | 2 | 0.22x | 0.009 |

- Condition-space gap ratio: **0.0%**
- Intrinsic dimensionality of visited states: **2.67**
- Feature shards: **not assessed** — no observation.image_features in this dataset; visual sharding was not assessed

## How this dataset was read

- Source format: **lerobot** (codebase v3.0)
- Phase labels: **gripper** — derived, not present in the dataset
- Trajectories compared in: **first-3-joints**
