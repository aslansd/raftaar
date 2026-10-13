# Raftaar report — `clean`

120 episodes · 16,800 frames · no training required


## 2 findings (0 critical)


### [INFO] ACTION_INCONSISTENCY — phase 'grasp', mostly on d_ee_y

**Observed.** different episodes in near-identical states disagree on 46% of this phase's action variance

**Consequence.** Either the operator was inconsistent, or the state is missing the variable that decides this. Check whether d_ee_y is a discrete decision the observation cannot see; if so, action chunking or an explicit phase input will help more than more data.


### [INFO] IDLE_CHANNEL — d_gripper in 'approach', d_gripper in 'transport'

**Observed.** no variance in this phase

**Consequence.** Usually by design -- a gripper is idle during an approach. Re-run with `--reference` against a dataset you trust to tell design from defect.


## Raw measurements


| phase | strategies | averaging hazard | H(a|s) |
|---|---|---|---|
| approach | 1 | 0.00x | 0.090 |
| grasp | 1 | 0.00x | 0.457 |
| transport | 1 | 0.00x | 0.092 |
| release | 1 | 0.00x | 0.004 |

- Condition-space gap ratio: **4.1%**
- Intrinsic dimensionality of visited states: **2.45**
- Feature shards: **1** (CKA between largest two: 1.00)
