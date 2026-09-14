# Changelog

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
