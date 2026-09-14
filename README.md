# Raftaar

**رفتار** — *behaviour.* The same word in Persian, Turkish and Urdu.

**Know what your demonstrations will teach, before you train.**

Raftaar reads a robot demonstration dataset and predicts *which policy class
will fail on it, and where* — without training anything. It runs on a CPU in
minutes.

Structural validators ("is this file corrupt, are the camera intrinsics valid")
are commoditising fast. Raftaar works one level up, on the **distribution**:
the geometry of the demonstrations themselves, and what a learning algorithm
will converge to when trained on them.

```bash
pip install raftaar
```

Python 3.10+. Apache 2.0. Three dependencies — numpy, scipy, scikit-learn. No
GPU, no torch, no downloads.

---

## The failure that matters

A dataset containing two valid but incompatible strategies — half the operators
reach around an obstacle on the left, half on the right — will train a
regression policy to do neither. The policy converges to the average, the
average is a path nobody demonstrated, and **the training loss goes down the
entire time**.

Validation loss does not catch it either, because the validation split has the
same structure as the training split. The same is true of coverage holes, of
silent recalibrations mid-collection, and of actuator channels that stopped
moving. Every one of these is detectable before training. None of them is
currently detected.

### Where this sits

The robot-data tooling layer is filling in from the bottom. Structural linters
that check schemas, timestamps and decodability already exist and are improving
fast — `trajlens` is one, and it audits the public LeRobot Hub today. That layer
is commoditising and Raftaar does not compete with it.

Raftaar works on the layer above: not "is this file valid" but **"what will a
policy learn from this distribution, and which policy class will fail"**. The
two are complementary — run a structural linter first, then this.

---

## Quickstart

```bash
# 1. A dataset with known, deliberately injected faults
raftaar synth field_data --episodes 120 \
    --faults bimodal_detour coverage_hole demonstrator_jitter camera_shard dead_actuator

# 2. Diagnose it — no training
raftaar scan datasets/field_data --out reports/field_data

# 3. Check the diagnosis was right — train policies and roll them out
raftaar validate datasets/field_data --trials 40
```

From Python:

```python
from raftaar import load_dataset, scan

report = scan(load_dataset("datasets/field_data"))
for finding in report["findings"]:
    print(finding["severity"], finding["id"], finding["where"])
    print("   ", finding["what"])
```

---

## Real datasets

Raftaar reads **LeRobotDataset** directories directly — the format published on
the Hugging Face Hub:

```bash
pip install "raftaar[lerobot]"
hf download lerobot/aloha_sim_transfer_cube_human --repo-type dataset \
    --local-dir ~/lerobot-data/aloha --exclude "videos/*"
raftaar scan ~/lerobot-data/aloha --max-episodes 50
```

Step by step, including where the datasets are and what to check first:
[RUNNING-ON-LEROBOT.md](RUNNING-ON-LEROBOT.md).

Both published layouts are read — **v2.x** (one episode per parquet) and
**v3.0** (many episodes per file). Episodes are separated on the
`episode_index` column, which exists in both.

```python
from raftaar import scan
from raftaar.lerobot import load_lerobot

report = scan(load_lerobot("path/to/dataset", max_episodes=50))
```

It reads the **format**, not the library. `lerobot` depends on torch,
torchvision and opencv; a CPU dataset audit should not pull a GPU stack to open
a parquet file. The only extra dependency is `pyarrow`.

### Two things real datasets do not have

Hub datasets carry no **phase labels** and no explicit notion of which state
columns are **spatial**. Raftaar derives both, and records how — because a scan
that guessed is not comparable to one that did not:

```
read as LeRobotDataset: 60 episodes, phases by gripper,
trajectories compared in first-3-joints
```

**Phases** come from the gripper channel: the moments it closes and opens are
the boundaries an operator actually works to. Without an identifiable gripper
the episode is cut into equal thirds and labelled `thirds`, which is a weaker
segmentation and says so.

**Spatial columns** are Cartesian when the feature names say so (`ee_x`, `ee_y`,
`ee_z`), and otherwise the first three joints. Comparing trajectories in joint
space is legitimate; calling joint angles a position is not, so the report says
which happened rather than leaving you to assume.

**Visual sharding is skipped**, not silently passed. No hub dataset ships
precomputed image features, so that detector reports `not assessed` with a
reason instead of "0 shards found" — a check that did not run is not a clean
result.

---

## The five detectors

| Detector | Question it answers |
|---|---|
| `AVERAGING_HAZARD` | Do demonstrations contain competing strategies whose *average* is a trajectory nobody performed? |
| `COVERAGE_GAP` | Which task conditions were never demonstrated? (sample-size aware — sparsity is not a hole) |
| `ACTION_INCONSISTENCY` | Do different episodes, in near-identical states, disagree on what to do? |
| `REPRESENTATION_SHARD` | Was part of this dataset recorded under a different calibration or lighting? |
| `IDLE_CHANNEL` | Which action channels carry no signal? (sharpened with `--reference`) |

`AVERAGING_HAZARD` is the differentiated one. It clusters demonstrations into
strategies, measures how far their average sits from the nearest real strategy
in units of within-strategy spread, and **suppresses splits that the task
conditions themselves explain** — paths to objects on the left and on the right
*should* differ, and that is the policy's job, not a defect.

---

## Why the naive version doesn't work

Three things had to be right before any of this detected anything:

1. **Strategy signatures.** Comparing raw trajectories fails: variation in
   *where the object was* swamps variation in *how the operator got there*.
   Subtracting the straight line between a phase's own endpoints isolates
   strategy from condition.
2. **A confound guard.** A mode split that the initial conditions predict is
   suppressed, or every well-collected dataset would be flagged.
3. **Elbow selection, not BIC minimum.** BIC falls monotonically as components
   are added, because real demonstration clusters always have sub-structure.

---

## Does it actually predict anything?

`raftaar validate` trains a unimodal regression policy (stands in for
ACT-style behaviour cloning) and a mode-conditioned policy (stands in for
diffusion/flow policies) on the same data and rolls both out. 120 episodes,
40 trials.

| Dataset | Raftaar verdict | Unimodal BC | Mode-conditioned BC |
|---|---|---|---|
| `clean` | no warnings | **100% success, 0% collision** | 78% success, 15% collision |
| `bimodal_only` | AVERAGING_HAZARD | 75% success, **25% collision** | **100% success, 0% collision** |
| `field_data` | 5 findings | 20% success, **80% collision** | 88% success, 5% collision |

The prediction runs both ways. On clean data the tool stays quiet and the simple
policy wins outright — reaching for a multimodal policy there makes things
*worse*. On bimodal data the tool warns, and the simple policy drives into the
obstacle. That symmetry is what makes the output actionable rather than
decorative.

### A caveat on small datasets

"No warnings" on clean data holds at 120 episodes. Below roughly 90,
`ACTION_INCONSISTENCY` fires at *warning* severity on clean data, because
near-identical states genuinely do disagree when there are few episodes per
region of the state space. The detector is measuring something real; it is
under-powered, not wrong. Both behaviours are pinned by tests in
`tests/test_detectors.py` so the boundary is documented rather than folklore.

---

## Optional extras

The core install is deliberately small. Everything else is opt-in:

```bash
pip install "raftaar[lerobot]"     # pyarrow, to read LeRobotDataset directories
pip install "raftaar[plot]"        # matplotlib, for the diagnostic figure
pip install "raftaar[validate]"    # the rollout study
pip install "raftaar[provenance]"  # record what produced each scan
pip install "raftaar[web]"         # the browser demo in web/
pip install "raftaar[all]"
```

`raftaar scan` writes its markdown report and JSON with no plotting stack
installed, and tells you what it skipped. A dataset audit should not require
matplotlib to run.

### Provenance

A scan is a measurement and `validate` is an experiment, so both depend on
seeds, library versions and the exact dataset that went in. With
[daftar](https://pypi.org/project/daftar/) installed:

```python
from raftaar.provenance import tracked_scan

report = tracked_scan("datasets/field_data", label="field-data-audit")
```

This records the dataset's content hash, the Raftaar version, every detector's
headline number and all findings, so two scans of the same dataset can be diffed
rather than eyeballed. Without daftar it behaves exactly like
`scan(load_dataset(path))` — the integration is a no-op, not an error.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 60 tests
```

The detector tests are the ones that matter. Every fault is *injected
deliberately* into the synthetic environment, so each detector is scored against
ground truth rather than opinion — and both directions are tested:

- **sensitivity** — a dataset built with a fault must raise that fault's finding
- **specificity** — clean data must stay quiet, because a detector that fires on
  everything is worse than no detector

---

## Status

Research prototype, honestly labelled.

The synthetic environment exists so the detectors can be scored against ground
truth, and the LeRobot adapter means they now run on data nobody generated to
order. What has **not** happened yet is the part that decides whether any of
this is useful: a validation study across a few dozen public datasets, training
small policies and testing whether the pre-training metrics predict
post-training success.

Until that exists, treat the numbers as a hypothesis with a working
implementation behind it.

## Licence

Apache 2.0.
