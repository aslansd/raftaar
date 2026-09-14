# Testing Raftaar

```bash
pip install -e ".[dev]"
pytest -q
```

Expect **60 passed**. Nothing should skip unless `pyarrow` or `matplotlib` is
absent.

---

## The three suites

| File | Tests | Needs | Covers |
|---|---|---|---|
| `tests/test_detectors.py` | 14 | nothing | the five detectors, scored against injected ground truth |
| `tests/test_packaging.py` | 17 | nothing | public API, CLI, reporting, optional dependencies |
| `tests/test_lerobot.py` | 29 | `pyarrow` | the LeRobotDataset adapter |

---

## Why the detector tests are the ones that matter

Every fault is **injected deliberately** into the synthetic environment, so each
detector is scored against something better than opinion: the dataset is known
to contain the defect, and the question is whether the tool says so.

Both directions are load-bearing:

- **Sensitivity** — a dataset built with `bimodal_detour` must raise
  `AVERAGING_HAZARD`; `coverage_hole` must raise `COVERAGE_GAP`; and so on.
- **Specificity** — clean data must stay quiet. A detector that fires on
  everything is worse than no detector, because it teaches users to ignore it.

The second direction is the one most projects skip, and it is the one that
decides whether anyone keeps using the tool after the first week.

### A documented limitation

`TestCleanDataSeverityIsSampleSizeDependent` pins something the README used to
overstate. "Clean data → no warnings" holds at 120 episodes:

| Episodes | Clean-data finding |
|---|---|
| 60 | `ACTION_INCONSISTENCY` at **warning** |
| 90 | `ACTION_INCONSISTENCY` at *info* |
| 120 | `ACTION_INCONSISTENCY` at *info* |

Near-identical states genuinely do disagree when there are few episodes per
region of the state space, so the detector is **under-powered rather than
wrong** — and the severity logic already demotes the finding as `n` grows. But
someone scanning 40 episodes sees a warning on data that is fine.

The test for the spurious warning carries a note: if it ever starts failing
because the warning no longer appears, that is an improvement and the test
should be deleted.

---

## The LeRobot adapter tests

These write the **LeRobotDataset v2.x layout on disk** — `meta/info.json`,
`meta/episodes.jsonl`, `meta/tasks.jsonl`, `data/chunk-000/*.parquet` — rather
than mocking the reader. The adapter tracks a file format, so the tests have to
exercise the file format.

Two robot shapes are covered because they are the two the adapter must tell
apart:

- a **joint-space arm** with a named gripper (the SO-100/SO-101 convention)
- a **Cartesian** dataset with `ee_x`/`ee_y`/`ee_z` columns

Getting the second wrong is the expensive failure. Treating joint angles as a
position produces numbers that look entirely plausible and mean something else,
and nothing downstream would catch it. `test_joint_space_falls_back_and_says_so`
exists for exactly that.

### The three properties worth preserving

1. **No torch.** `test_does_not_import_lerobot_or_torch` runs the adapter in a
   subprocess and asserts neither `lerobot` nor `torch` ends up in
   `sys.modules`. The adapter reads the format, not the library — that is why a
   CPU audit tool can open hub datasets at all.
2. **Derived phases are labelled as derived.** No hub dataset has phase labels.
   The adapter infers them from the gripper channel, falls back to equal thirds,
   and records which happened. A scan that segmented by thirds is not comparable
   to one that segmented by gripper, and nothing else would say so.
3. **An unrun check is not a clean result.** No hub dataset ships precomputed
   image features, so `representation_shards` reports `available: False` with a
   reason rather than "0 shards". `test_missing_image_features_are_reported_not_assumed`
   guards it, and the markdown report and the figure both say "not assessed".

---

## Running subsets

```bash
pytest tests/test_detectors.py -q          # fast, no optional deps
pytest tests/test_lerobot.py -v            # the adapter
pytest -k "clean" -v                       # specificity only
pytest -k "not lerobot" -q                 # if pyarrow is unavailable
```

---

## Trying it by hand

```bash
# Synthetic, with known faults
raftaar synth field --episodes 120 \
    --faults bimodal_detour coverage_hole camera_shard dead_actuator
raftaar scan datasets/field --out reports/field

# A real LeRobot dataset, if you have one downloaded
raftaar scan ~/.cache/huggingface/lerobot/<repo_id> --max-episodes 50
```

The format is detected from the directory: `data/episode_*.npz` is Raftaar's
own, `data/**/*.parquet` is LeRobot. Pass `--format` if detection is wrong.

Check the header line it prints:

```
read as LeRobotDataset: 60 episodes, phases by gripper,
trajectories compared in first-3-joints
```

Those two facts change what the findings mean. If it says `phases by thirds`,
the segmentation was a guess and the per-phase results are weaker.

---

## Before a release

```bash
pytest -q                                   # 60 passed
raftaar scan datasets/clean --out /tmp/r    # runs clean
python -m build && python -m twine check dist/*
```

Then install the wheel into a fresh virtualenv and invoke the CLI from there —
see `PUBLISHING.md`. It catches packaging faults the suite cannot.
