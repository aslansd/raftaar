# Raftaar — browser demo

A thin Flask wrapper over the library: choose which faults to inject, generate a
dataset, scan it, and read the findings next to the diagnostic figure. It exists
so someone can understand what Raftaar does in ninety seconds without installing
anything.

**Not part of the installable package.** `pip install raftaar` gives you three
dependencies — numpy, scipy, scikit-learn. Shipping Flask and gunicorn to
someone who wants to audit a dataset on a cluster would be wrong, so the demo
lives here and is installed through the `[web]` extra.

---

## Run it locally

From the **repository root**:

```bash
pip install -e ".[web,plot]"
python web/app.py
```

Then open <http://127.0.0.1:5000>.

Pick a fault or two, set an episode count, and submit. Each request builds a
synthetic dataset, scans it, renders the figure, and throws the dataset away.

---

## What it demonstrates

The point is the **pair of directions**, not either one alone:

- Inject `bimodal_detour` → `AVERAGING_HAZARD` fires, and the figure shows two
  strategy clusters with their mean sitting between them, in a region nobody
  demonstrated.
- Inject nothing → the tool stays quiet.

A detector that only ever warns is not a diagnosis, it is a decoration. Showing
both is what makes the demo persuasive to someone who has seen a lot of tools.

Use at least **80 episodes**. Below that the detectors are under-powered and
`ACTION_INCONSISTENCY` can fire on clean data — real behaviour, documented in
`TESTING.md`, but a poor first impression. The form enforces a floor for that
reason.

---

## Design notes

**Stateless by construction.** Every request builds its dataset in a fresh
temporary directory and deletes it. Nothing is written outside `/tmp`, so the
app runs unchanged on a read-only filesystem — Cloud Run, App Engine, any
container platform — and two users cannot interfere with each other.

**The figure is inlined as base64**, not served from disk, for the same reason:
there is no disk to serve it from.

**One worker, a few threads.** A scan is CPU-bound and takes seconds. Threads
handle the waiting; more workers would just multiply memory for no throughput.

---

## Deploy

The Dockerfile builds from the **repository root**, not from `web/`, because it
installs the library from source. That way the demo can never drift from the
code it demonstrates.

```bash
docker build -f web/Dockerfile -t raftaar-demo .
docker run -p 8080:8080 raftaar-demo
```

Cloud Run:

```bash
gcloud run deploy raftaar-demo \
    --source . \
    --region europe-west1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --timeout 120
```

1 GiB because scikit-learn plus matplotlib plus a few hundred episodes is
comfortably under it and 512 MiB is not. The 120-second timeout matches
gunicorn's.

---

## What it is not

This demo runs the **synthetic** environment only. It does not read
LeRobotDataset directories — that is the CLI's job, and uploading someone's
robot data to a web service is exactly the wrong shape for a tool whose whole
argument is that it runs locally on a laptop:

```bash
raftaar scan ~/.cache/huggingface/lerobot/<repo_id> --max-episodes 50
```

If the demo ever grows dataset upload, that argument goes with it.
