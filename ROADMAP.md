# Roadmap

## Where this stands

Published on PyPI, reading real LeRobotDataset directories, 60 tests.

| | Status |
|---|---|
| Five detectors, scored against injected ground truth | shipped |
| Synthetic environment with named faults | shipped |
| Validation harness (`validate`) | shipped |
| CLI, markdown + JSON + figure reports | shipped |
| LeRobotDataset adapter, v2.x **and** v3.0 | shipped |
| Browser demo | shipped |
| **Detectors calibrated against real data** | **not started — this is the gap** |

The first scans of five hub datasets ran without crashing, on the v3.0 format,
with zero episodes skipped. That is the months 1–3 milestone: the tool reads
real data.

It is not evidence the tool works.

---

## Read this before publishing any findings

**Four of the five datasets produced zero findings.** aloha_transfer,
aloha_insertion, pusht and so100 were all silent; droid raised one `info`.

That is not five clean datasets. Real teleoperated data is not clean — DROID
alone spans 76k episodes, dozens of operators and hundreds of scenes, and is
the canonical example of the heterogeneity this tool claims to detect.

The numbers say what happened:

| dataset | phase | modes found | hazard ratio |
|---|---|---|---|
| synthetic `bimodal_detour` | approach | 2 | **2.8x** ← fires |
| droid | approach | 3 | 0.53 |
| pusht | release | 2 | 0.75 |
| aloha_transfer | release | 2 | 0.15 |
| so100 | transport | 2 | 0.22 |

**The detectors are finding multiple strategies everywhere and calling none of
them hazardous.** The threshold was tuned on a synthetic fault built to be
detectable; on real data the same statistic lands three to twenty times lower.

Two possibilities, and they need to be told apart before any public claim:

1. The threshold is miscalibrated — real multimodality is subtler than the
   injected kind, and the tool is under-detecting.
2. The strategy signature is degenerate on real trajectories — note
   `hazard = 2.9e-17` for two `transport` phases, which is a numerical zero, not
   a measurement. That is the signature collapsing, and it is currently reported
   as a clean result.

Until that is resolved, **a quiet Raftaar report on a real dataset means
nothing**. Do not publish the five scans as evidence of anything.

---

## Priority 1 — Calibrate against reality

Nothing else matters until this is done.

**1.1 Find out whether `hazard ≈ 0` is a measurement or a failure.**
Two phases returned `2e-17`. Instrument `strategy_signature`: if the residual
after subtracting the endpoint-to-endpoint line has near-zero variance, the
phase is too short or too linear to carry a signature, and the honest output is
`insufficient signal`, not `0.00x`. Same class of bug as the `not assessed`
shard state — a number that reads as a clean result and is really a
non-measurement.

**1.2 Get ground truth on real data.** Take one dataset where you can inspect
the demonstrations, and label a handful of episodes by hand: which ones took a
visibly different route? Then ask whether the clustering recovers your labels.
Fifty episodes and an afternoon; without it, every threshold is a guess.

**1.3 Re-derive the threshold.** Once you know what a real multimodal dataset
scores, the threshold is an empirical question rather than a tuning parameter.
Report it as a distribution over the datasets you scanned, not a constant.

**1.4 Run `validate` on real data.** The harness exists and has never been
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
