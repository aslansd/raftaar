# Changelog

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
