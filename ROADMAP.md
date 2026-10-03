# Roadmap

## Where this stands

Published on PyPI, reading real LeRobotDataset directories, 88 tests.

| | Status |
|---|---|
| Five detectors, scored against injected ground truth | shipped |
| Synthetic environment with named faults | shipped |
| Validation harness (`validate`) | shipped |
| CLI, markdown + JSON + figure reports | shipped |
| LeRobotDataset adapter, v2.x **and** v3.0 | shipped |
| Calibration against a known grouping | shipped |
| Browser demo | shipped |
| **Detectors calibrated against real data** | **partly answered — see below** |
| **Does a warning predict a training failure** | **untested — the real claim** |

---

## The calibration result

The question was whether thresholds tuned on a synthetic generator mean anything
on the hub. It is now partly answered, on four multi-task datasets where the
task partition acts as free ground truth:

| dataset | tasks | best phase | ARI | ceiling | % of ceiling |
|---|---|---|---|---|---|
| `austin_sirius_dataset` | 2 | approach | 0.9435 | 1.0000 | **94.3 %** |
| `berkeley_rpt` | 4 | transport | 0.4033 | 0.5026 | **80.2 %** |
| `ucsd_pick_and_place_dataset` | 3 | transport | 0.0375 | 0.9960 | 3.8 % |
| `roboturk` | 3 | — | — | — | — (one mode in every phase) |

**Two of four transfer, strongly.** `austin_sirius_dataset` recovers the task
partition in all three phases (0.94 / 0.53 / 0.76), which rules out a lucky
phase. That is the first evidence the detectors measure something outside the
synthetic environment.

Three things it does not settle, all of which are now the roadmap:

1. **Every one ran with `phase_method: thirds`.** No gripper was identifiable in
   any of the four, so phase boundaries were guessed. A bad cut and an absence
   of structure are indistinguishable from the outside.
2. **`ucsd` finds two stable modes that align with nothing** — 3.8 % of a 0.996
   ceiling, so not a cardinality artefact. Those modes are real to the detector.
   Nobody knows what they are.
3. **`roboturk` finds no modes at all.** Its `names` are `motor_0 … motor_7`,
   the placeholder tier, so this is also the hardest dataset to interpret.

Earlier scans of `aloha_transfer`, `aloha_insertion`, `pusht`, `so100` and
`droid_1.0.1` were quiet or weak. `droid` is now explained: 1.93 episodes per
task and 59 collectors over 468 episodes leave almost no repetition to recover.

---

## Priority 1 — Explain the two failures

The positives are banked. The failures are now where the information is, because
each has a specific, testable cause.

**1.1 Phase segmentation. Velocity minima were tried and do not work.**

All four calibration datasets fell back to `thirds` because no gripper was
identifiable. That is the single confound running through every result.

Cutting at speed minima was the obvious fix and is measurably worse. Against
fixtures with known boundaries:

| true boundaries | velocity | thirds |
|---|---|---|
| (40, 85) | 0.889 | **0.958** |
| (25, 55) | **0.710** | 0.667 |
| (60, 100) | 0.643 | **0.667** |

0.747 against 0.764 on average. The cause is structural rather than tuning: an
episode decelerates at the end of *every* segment including the last, so the two
deepest interior minima are often the second and third boundaries instead of the
first and second. It found `[63, 97]` where the truth was `(25, 55)`. Not
shipped — a segmentation scoring below equal thirds makes every result harder to
interpret.

**Two directions left:**

- **Segment on the action signal rather than the state.** A phase change is
  often a change in *which channels are moving*, not a slowdown. Untried.
- **Do not segment at all.** Run the detector on whole episodes and see whether
  `austin_sirius` still separates at 94 %. If it does, phases were never
  load-bearing and this entire confound dissolves.

The second is the cheapest experiment available — a few hours — and informative
whichever way it goes. **Do it first.**

**1.2 Find out what `ucsd`'s two modes are.** They are stable across all three
phases and unrelated to the task, so they are not noise. Plot them. If they turn
out to be something like a left/right variant within a task, that is a genuine
finding about the dataset; if they are an artefact of the signature, that is a
detector fix.

**1.3 Get hand-labelled ground truth.** Take one dataset where you can inspect
the demonstrations, and label a handful of episodes by hand: which ones took a
visibly different route? Then ask whether the clustering recovers your labels.
Fifty episodes and an afternoon; without it, every threshold is a guess.

**1.4 Re-derive the threshold.** Once you know what a real multimodal dataset
scores, the threshold is an empirical question rather than a tuning parameter.
Report it as a distribution over the datasets you scanned, not a constant.

**1.5 Run `validate` on real data.** The harness exists and has never been
pointed at a hub dataset. It is the only thing that turns "the tool warns" into
"the warning was right", and it tests the claim the README actually makes.

Until it runs, the honest one-line summary of raftaar is *"finds strategy splits
that sometimes agree with task labels"*. That is real, and it is not the same
claim.

## Priority 2 — Adapter robustness, driven by what real data showed

**2.1 Feature names are often missing or non-standard.** Several datasets
publish none, so the adapter fell back to positional conventions. 0.3.1 stopped
it guessing a gripper when names exist and none matches — the PushT bug, where a
2-D pusher's *y coordinate* was treated as a gripper and phases were segmented
on it. 0.6.0 added `STATE_ALIASES` / `ACTION_ALIASES` for the `robomme` case,
where `state` / `actions` read as a dataset with no action column at all, and
records which names were resolved.

Keep finding these: each one is a dataset the tool silently misread. Across 180
datasets there are 129 distinct feature key names and only four are universal,
so this category is not nearly exhausted.

**2.2 Count how often `thirds` fires** across a larger sample. If phase
segmentation is guessed for most of the hub, per-phase detection is mostly
guesswork and the design needs revisiting, not the threshold.

**2.3 Handle bimanual properly.** Aloha is two arms in one 14-dimensional state
vector. The adapter takes the first three columns — one arm's first three
joints — and calls it the trajectory. Not wrong exactly, but it is analysing
half a robot without saying so.

## Priority 3 — Only after 1 and 2

- **Image features from video.** Decoding a handful of frames per episode and
  embedding them cheaply would make `REPRESENTATION_SHARD` work on hub data,
  where it is currently always `not assessed`. Real value, real cost: it breaks
  the CPU-only, no-opencv promise. Make it an extra, never a core dependency.
- **A `scan --compare` mode** over many datasets at once, producing one table.
  That is the artifact a blog post is built from.
- **Severity that scales with `n`.** `ACTION_INCONSISTENCY` fires at *warning*
  on clean data below ~90 episodes. It already demotes with sample size; the
  same treatment is probably owed to the other detectors.

---

## Priority 4 — Read what upstream now records

Three libraries have started recording values daftar's adapters reconstruct,
because of issues raised while writing those adapters:

| library | what it now records | adapter still reconstructs |
|---|---|---|
| cpm | `number_of_starts`, `initial_guess_supplied` (#85, merged) | `len(optimiser.initial_guess)` |
| sbi | `summary["converged"]` (#2014 → #2018, merged) | `epochs_trained[-1] < max_num_epochs` |
| MNE | `ICA.converged_` (#14366 merged, #14370 in review) | `n_iter_ < max_iter` |

Each reconstruction is subtly wrong against the library's own definition — sbi
uses `<=` where the adapter uses `<`, and MNE's `n_iter_` carried no convergence
information at all for infomax before #14366. A reconstructed value drifts from
the definition it imitates, silently, which is the thesis of the whole project
demonstrated on its own code.

`verify_three_upstreams.py` checks all three. Update each adapter to prefer the
library's answer and record which source was used, as the LeRobot adapter
already does with `gripper_source`.

## Explicitly not now

- **More detectors.** Five that are calibrated beat eight that are not, and
  adding a sixth before Priority 1 would just be more uncalibrated output.
- **A hosted service.** The argument is that it runs locally on your laptop.
  Uploading robot data to a web service contradicts it.
- **Training-time integration.** Raftaar's whole claim is that it tells you
  something *before* you train. A training-loop hook is a different product.

---

## The gate

Before writing anything public about real datasets, you should be able to say:

1. **This dataset has competing strategies, and here is the evidence.** Not a
   number below a threshold — a figure, and a couple of episodes you can point
   at that took visibly different routes.
2. **And this one does not**, by the same measure.
3. **A policy trained on the first fails in the way predicted**, from
   `validate`.

That is the claim the feasibility plan was built on, and the first two scans
where it holds are worth more than fifty scans where nothing fires.

---

## Honest status

Raftaar is a working implementation of a hypothesis that has not yet been
tested. The synthetic evaluation is real and the detectors do recover injected
faults — that is genuine, and more than most tools at this stage can show.

But the first contact with real data produced near-silence, and near-silence is
the outcome you would also get from a tool that does not work. Distinguishing
those two is the entire job right now.
