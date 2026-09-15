# Running Raftaar on real LeRobot data

## Where the datasets actually are

Your link was close. `huggingface.co/lerobot/collections` is the org's
collections tab; the datasets are at:

- **<https://huggingface.co/datasets/lerobot>** — the official org, ~181 datasets
- **<https://huggingface.co/datasets?other=LeRobot>** — everything tagged
  `LeRobot`, thousands, mostly community SO-100/SO-101 recordings
- **<https://huggingface.co/spaces/lerobot/visualize_dataset>** — browse one
  before downloading it

Start with the official org. Those are curated, well-formed, and the ones other
people will recognise when you publish findings.

---

## Step 1 — Install

```bash
pip install --upgrade "raftaar[lerobot,plot]"
pip install huggingface_hub
```

`[lerobot]` adds **pyarrow only**. It does not install the `lerobot` package,
which pulls torch, torchvision and opencv — Raftaar reads the dataset *format*
on disk, not the library.

## Step 2 — Download one dataset

```bash
hf download lerobot/aloha_sim_transfer_cube_human \
    --repo-type dataset \
    --local-dir ~/lerobot-data/aloha_transfer
```

(On older `huggingface_hub` the command is `huggingface-cli download` with the
same arguments.)

Check what you got:

```bash
ls ~/lerobot-data/aloha_transfer/meta/
cat ~/lerobot-data/aloha_transfer/meta/info.json | head -40
```

You want `codebase_version`, `robot_type`, `fps`, and the `features` block —
especially the `names` under `observation.state`. Those names are what the
adapter uses to find the gripper.

**Skip the videos if the dataset is large.** Raftaar never reads them:

```bash
hf download lerobot/<name> --repo-type dataset \
    --local-dir ~/lerobot-data/<name> \
    --exclude "videos/*"
```

## Step 3 — Scan it

```bash
raftaar scan ~/lerobot-data/aloha_transfer --max-episodes 50 --out reports/aloha
```

Start with `--max-episodes 50`. It is the difference between a thirty-second
look and a ten-minute one, and if the adapter mis-reads the dataset you want to
find out quickly.

## Step 4 — Read the header line before the findings

```
read as LeRobotDataset: 50 episodes, phases by gripper,
trajectories compared in first-3-joints
```

**This line decides what the findings mean.** Check both halves:

| What it says | What it means |
|---|---|
| `phases by gripper` | Good. Phase boundaries came from the gripper opening and closing. |
| `phases by thirds` | The adapter could not identify a gripper and cut each episode into equal thirds. Per-phase findings are much weaker. Look at `meta/info.json` → `features.observation.state.names` and see what the gripper channel is actually called. |
| `trajectories compared in cartesian-by-name` | The dataset exposes `ee_x`/`ee_y`/`ee_z`. Trajectories are compared in real space. |
| `trajectories compared in first-3-joints` | Joint-space arm. Comparing the first three joints is legitimate, but it is not a position — read `AVERAGING_HAZARD` as "these episodes moved the arm differently", not "these episodes took different paths through space". |

Then read `reports/aloha/report.md`, which repeats all of this under **How this
dataset was read**.

## Step 5 — Expect `not assessed` for shards

```
- Feature shards: **not assessed** — no observation.image_features in this dataset
```

Correct and expected. Hub datasets ship MP4 video, not precomputed image
features, so that detector cannot run. It says so rather than reporting "0
shards", which would read as a clean result from a check that never happened.

---

## What to do when it goes wrong

**`phase_method: thirds` when you expected `gripper`.** The most likely first
bug. Look at the state feature names:

```bash
python -c "
import json; f=json.load(open('$HOME/lerobot-data/aloha_transfer/meta/info.json'))
print(f['features']['observation.state'].get('names'))"
```

If the gripper is called something not in `raftaar.lerobot.GRIPPER_HINTS`
(currently gripper, grip, jaw, hand, claw, finger), add it. That is a one-line
PR to your own repo and exactly the kind of thing only real data reveals.

**`LeRobotFormatError: no episode parquet files`.** You pointed at `data/` or a
file. Point at the dataset root — the directory containing `meta/`.

**A scan that runs but reports one enormous episode.** Under v3.0 many episodes
share a parquet file and are separated by the `episode_index` column. Raftaar
splits on it, but if a dataset lacks that column it cannot. Check:

```bash
python -c "
import pyarrow.parquet as pq, glob
f = sorted(glob.glob('$HOME/lerobot-data/aloha_transfer/data/**/*.parquet', recursive=True))[0]
print(pq.read_table(f).column_names)"
```

**Everything fires at once.** Check the episode count first. Below ~90 episodes
`ACTION_INCONSISTENCY` fires on clean data — documented in `TESTING.md`. Raise
`--max-episodes` before concluding anything about the dataset.

---

## A suggested first batch

Pick five with different robots and collection styles, so adapter bugs surface
early rather than after you have scanned thirty of the same shape:

| Dataset | Why |
|---|---|
| `lerobot/aloha_sim_transfer_cube_human` | simulated, clean, bimanual — a sane baseline |
| `lerobot/aloha_sim_insertion_human` | same rig, harder task |
| `lerobot/pusht` | 2-D, tiny, fast; different action space entirely |
| an `so100`/`so101` community dataset | the single-arm convention the adapter targets; find one at `?other=LeRobot` |
| `lerobot/droid_1.0.1` | large and messy. Use `--max-episodes 200` and expect problems |

For each: record the header line, the findings, and anything the adapter got
wrong. That log is the raw material for the months 1–3 writeup.

---

## Honest expectations

I have not run this against a real hub dataset — no network access to Hugging
Face from where I built it. Every test is against fixtures written to the
published spec, for both v2.x and v3.0 layouts. **A fixture agrees with my
assumptions by construction; real data will not.**

So expect adapter bugs on the first two or three datasets. That is the point of
running it, and each one is a small, well-defined fix. What you should *not*
see is a scan that looks fine and is quietly wrong — the header line, the
`not assessed` shard state, and the `first-3-joints` label all exist so that the
interpretation is visible rather than assumed.

Once five datasets scan cleanly and you have read the reports, you have the
months 1–3 deliverable: **the tool runs on real data, and there is at least one
finding worth arguing about in public.**
