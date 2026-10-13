# Raftaar report — `pusht`

50 episodes · 6,235 frames · no training required


## No findings

No detector fired -- but this dataset was read with assumptions, so read that as *nothing was found* rather than *nothing is there*:

- phase labels were guessed (equal thirds), so per-phase results are weak
- visual sharding was not assessed


## Raw measurements


| phase | strategies | averaging hazard | H(a|s) |
|---|---|---|---|
| approach | 1 | 0.00x | 0.092 |
| transport | 1 | 0.00x | 0.098 |
| release | 1 | 0.00x | 0.078 |

- Condition-space gap ratio: **3.0%**
- Intrinsic dimensionality of visited states: **1.97**
- Feature shards: **not assessed** — no observation.image_features in this dataset; visual sharding was not assessed

## How this dataset was read

- Source format: **lerobot** (codebase v3.0)
- Phase labels: **thirds** — derived, not present in the dataset
- Trajectories compared in: **first-2-joints**
