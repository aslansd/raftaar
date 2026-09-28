# Next: roboturk

That comment is the one that unblocks you, and it does two things at once —
hands you the positive control you were missing, and explains why DROID was
never going to work.

**Why DROID fought you:** 49,630 tasks for 95,658 episodes — **1.93 episodes per
task** — and 59 `collector_id` values across your 468-episode sample. Both
groupings are nearly one-to-one with the episodes. A 2-mode detector had almost
no repetition to lock onto, and the ceiling correction cannot repair that. The
weak signal you reported was probably the best that comparison could have
produced.

---

## Install 0.5.0 first

`tasks` was not in `GROUPING_HINTS`, and it arrives as a **list of strings** per
episode, so `set(groups)` would have raised on it. Both fixed. Lists are
collapsed by joining rather than by taking the first element — two episodes with
the same *set* of tasks belong together.

**Validated end to end** on a fixture with three tasks and genuinely different
routes:

```
episodes=180, 3 tasks, 60 per task
  approach   modes=1  (tasks do not diverge here -- correct)
  transport  modes=3  ARI=+1.000  ceiling=1.000  frac=1.00
  release    modes=1  (correct)
```

Three modes recovered from three tasks, 100 % of ceiling. And the phases where
the tasks do not differ report one mode rather than splitting to match the
grouping — which is what you want, since a detector that always matched the
grouping would be useless.

There is also a test pinning the sample-size dependence: the same three tasks
and the same trajectory differences score **1.000 at 50 episodes/task and 0.56
at 20**. That is the quantitative form of their point.

81 tests.

```bash
pip install --no-cache-dir -U raftaar     # 0.5.0
```

## The run

`roboturk` first — 3 tasks, 1,995 episodes, 665 per task. Smallest clean test
they identified.

```bash
hf download lerobot/roboturk --repo-type dataset \
    --local-dir ~/lerobot-data/roboturk --exclude "videos/*"
```

```python
from raftaar import scan
from raftaar.lerobot import load_lerobot
from raftaar.calibration import calibration_report, episode_groups

data = load_lerobot("~/lerobot-data/roboturk", max_episodes=600, spread=True)
a = data["info"]["_adapter"]
print(a["sampling"], "|", a["phase_method"])
print("grouping keys:", a["grouping_keys"])         # expect 'tasks'
print("tasks:", len(set(episode_groups(data, "tasks"))))   # expect 3

report = scan(data)
print(calibration_report(data, report, "tasks"))
```

Then the same for `ucsd_pick_and_place_dataset` (3 tasks), `austin_sirius_dataset`
(2), `berkeley_rpt` (4). All are small.

## This test is decisive, which the others were not

You now know the method **can** recover a 3-way task partition, because it does
so at ARI 1.000 on a fixture. So:

| outcome on roboturk | what it means |
|---|---|
| recovers the 3 tasks | the method transfers. DROID was a bad test, not a bad method. |
| does not recover them | **your thresholds do not transfer.** Not "the hub is unimodal", not "the comparison was mismatched" — with 665 episodes per group and 3 coarse tasks, there is no excuse left. |

That second row is the honest null you promised in the original issue, and it is
now available in a form nobody can argue with. Say so plainly if you get it.

## Pre-register the prediction in the comment

Before you run it, write down what you expect. It costs one sentence and it is
the difference between a result and a rationalisation:

> Three tasks, 665 episodes each, and a detector that recovers three modes from
> three tasks at ARI 1.000 on synthetic data. If it cannot separate roboturk's
> tasks, the thresholds are the problem and I will say so.

## Also worth doing

**Thank them for the diagnosis, specifically.** The 1.93 episodes/task number
explains your own result better than you did, and they arrived at it by
scanning 189 datasets. That is the second time they have handed you something
more useful than what you asked for.

**Note their own bug**, briefly — they read `meta/tasks.jsonl`, which does not
exist, and reported "no task file" for all 189 datasets. *"Describing my reader,
not the hub."* That is now three instances of the same failure mode in this
thread across two people. If you ever write this up, that sentence is the
abstract.

**Still outstanding:** the `robomme` silent mis-parse. It has been open since
their first comment.
