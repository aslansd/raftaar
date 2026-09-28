# Raftaar × lerobot#4749 — the whole record

A single log of what happened on
[huggingface/lerobot#4749](https://github.com/huggingface/lerobot/issues/4749),
from opening the issue to posting the DROID ranking. Written so that in six
months you can reconstruct why each decision was made without re-reading eight
files.

---

## 1. The exchange

**You opened the issue** with three things: a torch-free reader for
LeRobotDataset, two format problems (no way to identify the gripper channel
generically; joint space vs task space not declared), and one request — does
anyone know a hub dataset that definitely contains two competing strategies?

That third question mattered because your detector was quiet on five hub
datasets, and *"the data is unimodal"* and *"my thresholds do not transfer"*
look identical from outside.

**They replied with a field-level audit of all 180 `lerobot/*` datasets.** The
substance:

| `names` quality | datasets | share |
|---|---|---|
| semantic (`left_gripper`, …) | 45 | 25.0 % |
| placeholder (`motor_0`, …) | 72 | 40.0 % |
| absent | 55 | 30.6 % |

- **129 distinct feature key names** across 180 datasets; only four universal
  (`timestamp`, `episode_index`, `index`, `task_index`).
- `robomme` uses `state`/`actions`/`image` — name-matching tooling reads it as
  having no action column at all.
- `droid_1.0.1` (35 features) vs `droid_100` (12) — same dataset, two naming
  systems.
- **`next.success` is dead**: of 180 datasets only 6 declare it; in the `pusht`
  family 3 of 4 are constant `False` while the trajectories demonstrably
  succeed.

Two corrections worth remembering, because both were right:

- *"Your 'last channel is the gripper' fallback was not the bug — the metadata
  layer is."*
- You proposed a new `semantic`/`role` field; they asked why the field that
  already exists for this, `features[k].names`, is meaningless in 75 % of the
  hub. The better question.

**On your third point they had no confirmed dataset**, but gave a lead that
turned out to be the most valuable thing in the thread: `droid_1.0.1` carries a
**`collector_id`** column.

**Then they published the readings** as
[`laa1991/lerobot-dataset-field-audit`](https://huggingface.co/datasets/laa1991/lerobot-dataset-field-audit) —
including `gripper_indices_state` for the 45 determinable datasets, the other
135 marked *not determinable* rather than guessed, nine reproduction scripts, and
a `findings/05` note stating that "standard names" is a set they chose.

---

## 2. Why `collector_id` changed the project

You could not distinguish "unimodal data" from "thresholds don't transfer"
because you had no partition to check against. A grouping key recorded **for
other reasons** is free ground truth for a weaker but decisive question:

> Where several operators demonstrably contributed, does the detector recover
> that partition at better than chance?

A detector that never aligns with any known grouping is not detecting
strategies. One that does is measuring something, and its silence elsewhere can
then be believed.

That became `raftaar.calibration`, validated on the synthetic generator before
touching real data:

```
positive control (generator's own injected labels)   ARI = +1.000
negative control (random grouping, same split)       ARI = −0.008
```

---

## 3. Versions, and what forced each one

| version | change | forced by |
|---|---|---|
| **0.4.0** | `raftaar.calibration`; `averaging_hazard` returns `mode_labels`; adapter carries per-episode metadata; `load_dataset` reads `meta/episodes.jsonl` back | the `collector_id` lead |
| **0.4.1** | `raftaar.field_audit` reads their CSV; gripper index taken from the audit where available | their published dataset |
| **0.4.2** | `spread=True` — `max_episodes` drew evenly across files instead of reading a prefix | 95,658 episodes across 168 files |
| **0.4.3** | string columns retained by the parquet reader | the first DROID run |
| **0.4.4** | `agreement()` reports the achievable **ceiling** alongside each score | the second DROID run |

77 tests at 0.4.4.

---

## 4. Five bugs, all mine, all the same shape

Worth keeping together, because the pattern is the point — every one produced
confident output describing the **tool** rather than the **data**, and none of
them raised an error.

1. **`max_episodes` read a prefix, not a sample.** On 95,658 episodes across 168
   files, the first 500 come from one or two files — one session, one or two
   operators. The grouping would have collapsed to a single level and
   `calibration_report` would have returned `usable: False`, reading as *"no
   structure found"* when the truth was *"no structure could have been found"*.
   Measured on a fixture with one collector per file: prefix gave 2 collectors,
   spread gave 12, on the same budget.

2. **String columns were silently dropped.** The parquet reader kept numeric and
   list columns, with the comment *"strings and nested structs are metadata; the
   detectors do not use them"*. True of the detectors, false of everything else:
   `collector_id`, `building` and `task_category` are all strings. The first
   DROID run reported `grouping_keys: ['task_index']` and *"no per-episode
   `collector_id` in this dataset"* — describing the reader, not DROID. You very
   nearly posted that to the person who had just published the audit.

3. **A mis-indented line broke v3.0 episode splitting** while wiring the audit
   in — 30 episodes across 3 files read as 3. Caught immediately by the v3 tests,
   which is what they were for.

4. **Raw ARI was the wrong headline statistic.** Comparing 2 detected modes
   against 59 operator ids compresses ARI severely; reporting it unqualified
   would have turned a directional result into a null one.

5. **My first ceiling estimate was wrong.** I hand-computed it with uniformly
   random groups and got ~0.033, giving "71 % of ceiling". The harness computes
   it from the real, unbalanced group sizes: 0.1654, giving **14 %**. I should
   have used the tool's own number rather than a back-of-envelope one.

Same shape as `next.success` in their audit: the schema says one thing, the code
says another, nothing errors.

---

## 5. The DROID result

468 episodes, spread across all 168 files. 59 `collector_id` values, 49
buildings. `phase_method: gripper` — DROID is one of the 45 with semantic names,
so per-phase numbers are trustworthy here.

| grouping | groups | best ARI | ARI ceiling | % of ceiling | AMI % of ceiling |
|---|---|---|---|---|---|
| `collector_id` | 59 | 0.0235 | 0.1654 | **14.2 %** | **8.1 %** |
| `building` | 49 | 0.0060 | 0.1031 | 5.8 % | 4.4 % |
| `task_category` | 49 | 0.0060 | 0.1031 | 5.8 % | 4.4 % |

**Reading:** `collector_id` is roughly 2× the controls on both statistics and in
every phase — more than expected — but 8–14 % of achievable is not a result.
A weak signal in the right direction.

**Two things it does not settle:**

- The detector found **2 modes in every phase**. It was not quiet. Whether that
  is real structure that happens not to follow operator, or an artefact of a
  clustering step that tends to return two, is unresolved.
- `building` and `task_category` returned **byte-identical** statistics at every
  phase. Verified not to be a reader bug (a fixture with differing columns keeps
  them distinct), so in DROID these two fields partition episodes the same way.
  That means the intended positive control was not independent, and the question
  above has no answer yet.

---

## 6. Still open

1. **`behavior1k-*` with `observation.task_info`** — now the only remaining
   positive control. Different subtasks genuinely are different trajectories. If
   the method cannot separate those, the problem is the thresholds rather than
   the hub, and on current evidence that is the likelier answer.
2. **The `building` / `task_category` identity** — one line settles whether they
   carry the same values or merely coincide. Either way it is a contribution to
   their audit.
3. **The `robomme` silent mis-parse** — `state`/`actions`/`image` produces an
   empty scan instead of an error. Handed to you with a reproduction, still
   unfixed.
4. **Recalibration.** Real demonstrations are far more locally consistent than
   the synthetic generator assumes — conditional action entropy 0.009–0.156 on
   hub datasets against 0.004–0.457 synthetic. A detector tuned on noisier data
   than it will meet is going to be quiet.

---

## 7. What worked, as method

Worth naming, because it is repeatable:

- **Lead with something useful to the reader.** Every message that got a reply
  carried a finding, a bug, or a capability. The one that carried only an
  announcement — the emails to friends — got nothing.
- **Post the controls with the result.** The synthetic +1.000 / −0.008 pair is
  what makes a weak number interpretable rather than suspicious.
- **Name your own bugs first.** Five of them here. It costs nothing once fixed
  and it is why the numbers can be trusted.
- **A control you expect to fail is worth more than the test itself.**
  `building` is the only reason the `collector_id` number means anything.
- **Check the ceiling before reading a ratio.** Chance correction is not enough
  when cardinalities differ by an order of magnitude.

---

## 8. Landscape note

Three packages now occupy the layer below Raftaar: `trajlens` (structural lint
and repair for LeRobot), `deepen-grade`, `robot-data-audit`. None does
distributional analysis, and the distinction — dataset hygiene versus *what a
policy will learn from this distribution* — is not obvious from outside.

Whatever you publish from this work should make that distinction in its first
paragraph, or it will be read as one more dataset linter.

---

```bash
hf download lerobot/droid_1.0.1 --repo-type dataset \
    --local-dir ~/lerobot-data/droid --exclude "videos/*"
```

```python
from raftaar import scan
from raftaar.lerobot import load_lerobot
from raftaar.calibration import calibration_report, episode_groups

data = load_lerobot("~/lerobot-data/droid", max_episodes=500, spread=True)
a = data["info"]["_adapter"]
print(a["sampling"], "|", a["phase_method"])
print("grouping keys:", a["grouping_keys"])       # expect collector_id, building
print("collectors:", len(set(episode_groups(data, "collector_id"))))

report = scan(data)
print(calibration_report(data, report, "collector_id"))
print(calibration_report(data, report, "building"))
print(calibration_report(data, report, "task_category"))
```
