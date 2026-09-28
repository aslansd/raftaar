# Roadmap

## Where this stands

Published on PyPI, reading real LeRobotDataset directories, 84 tests.

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

**1.1 Better phase segmentation than equal thirds.** All four calibration
datasets fell back to `thirds` because no gripper was identifiable. That is the
single confound running through every result, positive and negative. A
segmentation derived from the movement itself — velocity minima, dwell
detection — would cost little and would let the failures be attributed.

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
"the warning was right".

## Priority 2 — Adapter robustness, driven by what real data showed

**2.1 Feature names are often missing.** Several datasets published none, so the
adapter fell back to positional conventions. 0.3.1 stops it guessing a gripper
when names exist and none matches — the PushT bug, where a 2-D pusher's *y
coordinate* was treated as a gripper and phases were segmented on it. Keep
finding these: each one is a dataset the tool silently misread.

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
