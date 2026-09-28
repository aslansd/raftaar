# Changelog

## 0.5.0 — the task grouping

`meta/episodes/*.parquet` carries a **`tasks`** column: which task each episode
belongs to. It is per-episode, semantically bound to separate trajectories, and
independent of recording conditions — which is exactly what `collector_id` and
`building` were not. It is now the first entry in `GROUPING_HINTS`.

It arrives as a **list of strings** per episode, so `set(groups)` would have
raised on it. List values are collapsed by joining rather than by taking the
first element: two episodes labelled with the same *set* of tasks belong
together, and an episode with two tasks is not the same group as one with the
first of them.

### Why this matters more than `collector_id`

`droid_1.0.1` has 49,630 tasks for 95,658 episodes — **1.93 episodes per task** —
and 59 `collector_id` values. Both groupings are nearly one-to-one with the
episodes, so a 2-mode detector has almost no repetition to lock onto and the
ceiling correction cannot repair it.

The datasets worth testing are the ones with **few groups and many episodes
each**: `roboturk` (3 tasks, 665 episodes/task), `ucsd_pick_and_place_dataset`
(3, 452), `austin_sirius_dataset` (2, 280), `berkeley_rpt` (4, 227).

### Validation

On a fixture with three tasks and genuinely different routes, the detector
recovers the task partition at **ARI = +1.000**, 100 % of ceiling, with three
modes found from three tasks. Phases where the tasks do not diverge correctly
report a single mode rather than splitting to match the grouping.

Recovery is sample-size dependent and there is now a test that pins it: the same
three tasks and the same trajectory differences score 1.000 at 50 episodes per
task and 0.56 at 20. That is the quantitative form of why DROID was a poor test.

81 tests.

## 0.4.4 — report the ceiling, not just the score

Running the harness on `lerobot/droid_1.0.1` produced ARI = 0.0235 against
`collector_id`, which reads as "near zero, no alignment". It is not.

Chance correction is not enough when the two partitions have very different
cardinalities. Two detected strategies compared against **59** operator ids
cannot score highly however good the detector is — there is no 2-way split of 59
groups that agrees well with the 59-way partition, and ARI is compressed almost
to nothing:

| | ARI | ceiling | % of ceiling |
|---|---|---|---|
| `collector_id` (59 groups) | 0.0235 | 0.0332 | 71% |
| `building` (49 groups) | 0.0060 | 0.0426 | 14% |

`agreement()` now reports the **ceiling** each statistic could reach given the
cardinalities, plus the score as a fraction of it. AMI is far less compressed
than ARI and is the better headline when the cardinalities differ, so
`calibration_report` gained `best_ami_fraction_of_ceiling` alongside the raw
best ARI.

The previous headline — raw `best_adjusted_rand` — was the wrong statistic for
this comparison, and it would have made a directional result look like a null
one.

## 0.4.3 — string columns were being dropped

`_load_table` kept numeric and list columns and discarded everything else, with
the comment *"strings and nested structs are metadata; the detectors do not use
them"*. That was true of the detectors and false of everything else.

`collector_id`, `building` and `task_category` are strings. They are the only
labels on the hub that can act as ground truth for a detected strategy split,
and they were being thrown away before anything could see them.

The failure was silent and pointed the wrong way. On `lerobot/droid_1.0.1` the
adapter reported:

```
grouping_keys : ['task_index']
collector_id  : available=False, reason="no per-episode 'collector_id'"
```

`task_index` is an integer and survived; the three string columns did not. Read
at face value that says DROID has no operator labels. It has them — 
`calibration_report` was describing the reader, not the dataset.

String columns are now retained as `dtype=object`. Nested structs are still
skipped: no current use, and they do not flatten to anything a grouping key
could be.

Three tests pin it, including that numeric columns are unaffected.

## 0.4.2 — a prefix is not a sample

`max_episodes` read the **first** N episodes. On `lerobot/droid_1.0.1` — 95,658
episodes across 168 files — `max_episodes=300` reads roughly the first 0.3%,
which comes from one or two files. Hub datasets are written in collection order,
so that is one session by one or two operators.

For a smoke test that is fine. For comparing a detected split against
`collector_id` it is fatal in a way that does not look like a bug: the grouping
collapses to a single level, `calibration_report` returns `usable: False`, and
the result reads as "no structure found" when it is really "no structure could
have been found".

**`load_lerobot(..., spread=True)`** draws the same budget evenly across every
file instead. On a fixture with one collector per file:

```
spread=False   36 episodes,  2 distinct collectors
spread=True    36 episodes, 12 distinct collectors
```

It uses the floor rather than the ceiling of `max_episodes / n_files`, so it may
return slightly fewer episodes than asked in exchange for every file
contributing. Rounding up exhausts the budget before the last files are reached,
which drops the tail of the dataset — the opposite of the point.

`_adapter.sampling` records which mode was used, so a scan says whether its
episodes were a prefix or a spread.

## 0.4.1 — read the audit instead of guessing

A hub-wide audit of `features[k].names` was published at
[`laa1991/lerobot-dataset-field-audit`](https://huggingface.co/datasets/laa1991/lerobot-dataset-field-audit):
25% of datasets have semantic names, 40% have placeholders (`motor_0`, …), 30%
have none. It resolves the gripper channel for the datasets where that is
determinable and marks the rest as *not determinable* rather than guessing.

**New `raftaar.field_audit`.** Where the audit has an answer, the LeRobot
adapter uses it in preference to matching names against `GRIPPER_HINTS`; where
it does not, inference proceeds as before. The manifest records which happened
(`_adapter.gripper_source`), so a scan says whether the gripper was read or
inferred.

The reader is deliberately tolerant: the audit is not ours to version, so
renamed columns, a missing file or an unparseable shape are reported through
`missing_columns` and the audit behaves as though it knows nothing. A reader
that hard-failed on a renamed column would be worse than no reader.

### Fixed

An indentation error introduced while wiring the audit in broke episode
splitting for v3.0 layouts -- 30 episodes across 3 files were read as 3. Caught
by the existing v3 tests, which is what they were written for.

## 0.4.0 — calibration against known groupings

Raftaar could not distinguish "this dataset is unimodal" from "these thresholds
were tuned on a synthetic generator and do not transfer". From outside, those
look identical, and hand-labelling trajectories to settle it is expensive.

Some datasets carry a partition recorded for other reasons — `droid_1.0.1`
labels each episode with a `collector_id`, `behavior1k-*` carries
`observation.task_info`. Those are free ground truth for a weaker but decisive
question: **where several producers demonstrably contributed, does the detector
recover that partition at better than chance?** A method that never aligns with
any known grouping is not detecting strategies.

**New `raftaar.calibration`.** `agreement()` compares a detected split against a
known one using adjusted Rand and adjusted mutual information, both
chance-corrected. `calibration_report()` does it per phase.

It reports `usable` and a `reason` alongside the scores, because a comparison
that could not be made ("the detector found one strategy") must not read as one
that was made and failed. A null result and a negative result are different
findings.

**`averaging_hazard` returns `mode_labels`** — which episode was assigned to
which strategy. It was already computed and then discarded.

**The LeRobot adapter carries per-episode metadata.** Columns matching
`GROUPING_HINTS` (`collector_id`, `operator_id`, `session_id`, `task_index`,
`task_category`, `building`, …) are passed through as `episode["meta"]`. No
detector uses them; they exist to be checked against.

**Fixed: the synthetic generator's ground truth was unreachable.** `build_dataset`
wrote each episode's injected strategy to `meta/episodes.jsonl`, and
`load_dataset` never read it back. The generator's own labels are now available,
which is what makes the harness testable:

```
positive control (generator's strategy labels)   ARI = +1.000
negative control (random grouping, same split)   ARI = -0.008
```

Four new tests covering both controls, the unusable case, and a missing key.

## 0.3.1 — first contact with real data

Five hub datasets scanned. All read cleanly on the v3.0 format with zero
episodes skipped, which is the months 1–3 milestone. Three bugs surfaced that
only real data could have shown, all of the same kind: **the tool stating a
conclusion it had not earned.**

### It no longer invents a gripper

PushT is a 2-D pushing task with no gripper and no joints. The adapter fell back
to "last channel is the gripper" — an SO-100/Aloha convention — picked the
agent's **y coordinate**, segmented phases on it, and reported *Phase labels:
gripper* as fact.

Now: if feature names are published and none of them is a gripper, that is an
answer rather than a reason to guess. The positional fallback applies only when
no names exist at all, and only for action spaces of four dimensions or more,
because a 2-D action space is not an arm and there is nothing for the convention
to be true about. Phases then fall back to `thirds`, which the report says.

### "Train with confidence" is gone

A no-findings report used to end with that sentence. On PushT it appeared under
a scan that had misidentified the gripper, mislabelled the columns as joints,
and skipped visual sharding entirely.

A quiet report now states what was assumed to produce it — guessed phases,
unavailable feature names, unassessed sharding — so *nothing was found* is not
read as *nothing is there*. Where no assumptions were needed, it still notes
that the thresholds are calibrated against the synthetic environment.

### Labels no longer claim knowledge they lack

`first-3-joints` was reported for datasets that publish no feature names, where
the columns could be anything. Those now read
`first-N-state-dims (names unavailable)`.

### Added

`ROADMAP.md`, written against the scan results rather than a plan. Its first
priority is the thing those results exposed: four of five datasets produced zero
findings, with hazard ratios three to twenty times below the synthetic
threshold, and two phases returning a numerical zero that is a failure to
measure rather than a clean result.

## 0.3.0 — LeRobotDataset v3.0

### Reads the v3.0 layout

`LeRobotDataset:v3.0` packs **many episodes into a single parquet file**, where
v2.x wrote one episode per file. 0.2.0 read v2.x only, and on a v3 dataset it
raised `LeRobotFormatError: no episode parquet files` — a clean failure rather
than a wrong answer, but a dataset it could not open.

Episodes are now separated on the **`episode_index` column**, which exists in
both layouts, so the reader no longer depends on file naming at all. That was
the second bug: discovery globbed `data/**/episode_*.parquet`, and v3 names its
files `file-0000.parquet`, so the glob found nothing. It now matches any parquet
under `data/` and lets the column do the work.

Writing the v3 fixture is what caught this. The unit tests passed against a v2
fixture throughout.

### `--max-episodes` now counts episodes, not files

**A behaviour change**, and the reason this is 0.3.0 rather than 0.2.1. Under
v2.x, slicing the file list and counting episodes were the same thing. Under v3
one file can hold forty episodes, so `--max-episodes 10` would have read forty.
The limit is now applied while reading.

### Web template

The masthead is split across a tag for the two-tone colour:

```html
<div class="brand">Raf<span>taar</span></div>
```

so the product name never appears as a contiguous string, and every rename
searched for `DemoScope` and missed it. The `<title>` was contiguous and updated
correctly — which is why the browser tab read *Raftaar* while the page still
read *DemoScope*. There is now a test that strips tags before comparing, since
substring matching cannot catch this.

### Added

`RUNNING-ON-LEROBOT.md` — where the datasets actually are, how to download
without the videos, and what the scan header line means before you read any
findings.

60 tests, up from 53.

## 0.2.0 — real datasets

Months 1–3 of the plan: the library reads **LeRobotDataset** directories, so the
detectors run on data nobody generated to order.

### The adapter reads the format, not the library

`raftaar.lerobot.load_lerobot()` parses `meta/info.json`, `meta/episodes.jsonl`,
`meta/tasks.jsonl` and `data/**/*.parquet` directly. It does **not** import
`lerobot`, which depends on torch, torchvision and opencv — a CPU dataset audit
should not pull a GPU stack to open a parquet file. The only new dependency is
`pyarrow`, behind the `[lerobot]` extra, and a test asserts in a subprocess that
neither `torch` nor `lerobot` ends up in `sys.modules`.

### Two things hub datasets do not have

**Phase labels.** Derived from the gripper channel — the moments it closes and
opens are the boundaries an operator works to. Without an identifiable gripper,
the episode is cut into equal thirds. Which method ran is recorded per dataset
and printed by the CLI, because a scan segmented by thirds is not comparable to
one segmented by gripper and nothing else would say so.

**A notion of which state columns are spatial.** `averaging_hazard` previously
hard-coded `state[:, :3]` — true for the synthetic robot, false for a joint-space
arm, and wrong in the expensive way: it produces plausible numbers that mean
something else. The columns are now chosen by the adapter (Cartesian when the
feature names say so, first three joints otherwise), recorded in the manifest as
`spatial_dims`, and reported as `cartesian-by-name` or `first-3-joints`.

### An unrun check is no longer reported as a clean result

No hub dataset ships precomputed image features, so `representation_shards` now
returns `available: False` with a reason rather than "0 shards". The markdown
report says **not assessed**, and the figure's fourth panel says so too instead
of plotting an invented scatter.

The same applies to the figure's labels: the axes now name the actual channels
(`shoulder_pan`, `wrist_flex`) rather than claiming metres, and the synthetic
environment's obstacle marker is drawn only for synthetic data. A phantom
obstacle on a real dataset would be a fabricated hazard.

### CLI

```bash
raftaar scan <dataset> --max-episodes 50     # format auto-detected
raftaar scan <dataset> --format lerobot
```

Detection uses the data files, since both layouts carry `meta/info.json`:
`data/episode_*.npz` is Raftaar's own, `data/**/*.parquet` is LeRobot. A
directory that is neither now fails with a message naming both, rather than a
numpy traceback about concatenating an empty list.

### Docs

`PUBLISHING.md` and `TESTING.md` added, including the PyPI name-similarity trap
that blocked `raftar` (the name was unregistered *and* unregistrable, because
`rafter` exists). `web/README.md` rewritten, and the web Dockerfile fixed — it
referenced `requirements.txt` and a top-level `raftaar/` directory, neither of
which exists in the packaged layout, so the image could not have built.

### Tests

53, up from 29. The 24 new ones write the LeRobotDataset layout on disk and read
it back, covering both a joint-space arm with a named gripper and a Cartesian
dataset — the two shapes the adapter has to tell apart.

## 0.1.0

Named **Raftaar** — رفتار, *behaviour*, in Persian, Turkish and Urdu.

Behaviour cloning is the paradigm this tool serves, so the name says what the
library is about in one word to anyone who knows the field, and pairs with
[daftar](https://pypi.org/project/daftar/) (دفتر, *ledger*) as a second package
by the same author.

Renamed twice before release, from DemoScope via KineLens. "Demo" reads as a
demonstration *of a product* rather than demonstration *data*.

Deliberately not named `trajlint`, `trajscope` or `trajaudit`, all of which were
free: `trajlens` already exists on PyPI as a structural linter for LeRobot
datasets, and those names would read as derivative of a tool operating one layer
below this one.

`rename.py` is kept in the repository until first publication, so the name stays
cheap to change. It detects the current package name from `src/` rather than
hardcoding it, so it works however many times it is run.

## Packaging work (pre-release)

Raftaar existed as a working prototype: five detectors, a synthetic
environment with injected faults, a validation harness, a CLI and a browser
demo. This release turns it into something installable with `pip install
raftaar` without changing what the detectors do.

### Packaging

- `src/` layout, `pyproject.toml`, Apache 2.0 licence, `raftaar` console
  script replacing `python -m raftaar.cli`.
- **The dependency surface is now three packages**: numpy, scipy,
  scikit-learn. The prototype's `requirements.txt` also pulled matplotlib,
  Flask and gunicorn — a web server and a plotting stack, for a CPU dataset
  audit. Those are now the `[plot]` and `[web]` extras.
- `matplotlib` is imported lazily inside `report.plot()`. `raftaar scan`
  writes its JSON and markdown on a machine without it and prints what it
  skipped, instead of failing after the scan you just waited for.
- The Flask demo, its templates and the Dockerfile moved to `web/`, outside the
  installable package. They are the Cloud Run demo, not the library.

### Tests

29 tests, where there were none. The detector tests are the point: every fault
is injected deliberately into the synthetic environment, so each detector is
scored against ground truth, in both directions — sensitivity (an injected
fault must be caught) and specificity (clean data must stay quiet).

Also pinned: the CLI round-trip, the public API, that importing `raftaar`
pulls neither matplotlib nor Flask nor torch, and that the synthetic
environment is deterministic under a fixed seed — which it has to be, since it
is the ground truth everything else is scored against.

### A documented limitation

Writing the specificity tests surfaced something the README had overstated.
"Clean data → no warnings" holds at 120 episodes, but below roughly 90
`ACTION_INCONSISTENCY` fires at *warning* severity on clean data. Near-identical
states genuinely do disagree when there are few episodes per region of the state
space, so the detector is under-powered rather than wrong — but a user scanning
40 episodes sees a warning on data that is fine.

Both behaviours are now pinned by tests, and the README says so. The test for
the spurious warning carries a note that if it ever starts failing because the
warning no longer appears, that is an improvement and the test should be
deleted.

### Optional provenance

`raftaar.provenance` records what produced a scan — dataset content hash,
Raftaar version, every detector's headline number — using
[daftar](https://pypi.org/project/daftar/) when installed. Entirely optional:
`available()` is checked before anything is imported, `track()` yields `None`
when daftar is absent, and `tracked_scan()` returns exactly what
`scan(load_dataset(path))` returns either way.
