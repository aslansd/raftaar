"""
Raftaar web app.

A thin Flask wrapper over the library: choose which faults to inject, generate a
dataset, scan it, and read the findings next to the diagnostic figure.

Deliberately stateless. Every request builds its dataset in a fresh temporary
directory and throws it away, so the app runs unchanged on a read-only
filesystem (Cloud Run, App Engine, any container platform).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path

from flask import Flask, render_template, request

from raftaar.metrics import scan
from raftaar.report import plot
from raftaar.synth import DatasetSpec, FaultSpec, build_dataset, load_dataset

app = Flask(__name__)

MIN_EPISODES = 80
MAX_EPISODES = 200
DEFAULT_EPISODES = 120

FAULTS = [
    ("bimodal_detour", "Competing strategies",
     "Half the operators route around the obstacle one way, half the other."),
    ("coverage_hole", "Coverage hole",
     "A wedge of object positions is never demonstrated."),
    ("demonstrator_jitter", "Operator inconsistency",
     "Actions during transport vary between episodes for no visible reason."),
    ("camera_shard", "Mid-collection recalibration",
     "A subset of episodes is recorded with a different camera setup."),
    ("dead_actuator", "Stuck actuator",
     "The vertical channel stops responding while the object is carried."),
]

PRESETS = {
    "clean": [],
    "bimodal": ["bimodal_detour"],
    "field": [f[0] for f in FAULTS],
}

# Measured on 120-episode datasets, 40 rollout trials per cell. Rolled out
# ahead of time because training takes minutes and a web request should not.
VALIDATION = {
    "clean": {
        "verdict": "no warnings",
        "rows": [("Unimodal BC (MSE)", "100%", "0%"),
                 ("Mode-conditioned BC", "78%", "15%")],
    },
    "bimodal": {
        "verdict": "averaging hazard",
        "rows": [("Unimodal BC (MSE)", "75%", "25%"),
                 ("Mode-conditioned BC", "100%", "0%")],
    },
    "field": {
        "verdict": "5 findings",
        "rows": [("Unimodal BC (MSE)", "20%", "80%"),
                 ("Mode-conditioned BC", "88%", "5%")],
    },
}


def _run(selected: list[str], n_episodes: int) -> dict:
    """Build, scan and plot a dataset inside a throwaway directory."""
    work = Path(tempfile.mkdtemp(prefix="raftaar-"))
    try:
        spec = DatasetSpec(
            name="scan",
            n_episodes=n_episodes,
            faults=FaultSpec(**{f: True for f in selected}),
            seed=0,
        )
        path = build_dataset(spec, work)
        data = load_dataset(path)
        report = scan(data)

        fig_path = work / "report.png"
        plot(data, report, fig_path)
        figure = base64.b64encode(fig_path.read_bytes()).decode()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in report["findings"]:
        counts[f["severity"]] += 1
    report["counts"] = counts
    report["figure"] = figure
    return report


@app.get("/")
def index():
    return render_template("index.html", faults=FAULTS, selected=[],
                           episodes=DEFAULT_EPISODES, report=None,
                           validation=None)


@app.post("/")
def run_scan():
    selected = [f for f in request.form.getlist("faults")
                if f in dict((k, v) for k, v, _ in FAULTS)]
    try:
        episodes = int(request.form.get("episodes", DEFAULT_EPISODES))
    except ValueError:
        episodes = DEFAULT_EPISODES
    episodes = max(MIN_EPISODES, min(MAX_EPISODES, episodes))

    report = _run(selected, episodes)

    key = next((k for k, v in PRESETS.items() if sorted(v) == sorted(selected)), None)
    return render_template("index.html", faults=FAULTS, selected=selected,
                           episodes=episodes, report=report,
                           validation=VALIDATION.get(key))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/scan")
def api_scan():
    """JSON endpoint. ?faults=bimodal_detour,coverage_hole&episodes=120"""
    raw = request.args.get("faults", "")
    valid = {k for k, _, _ in FAULTS}
    selected = [f for f in raw.split(",") if f in valid]
    try:
        episodes = int(request.args.get("episodes", DEFAULT_EPISODES))
    except ValueError:
        episodes = DEFAULT_EPISODES
    report = _run(selected, max(MIN_EPISODES, min(MAX_EPISODES, episodes)))
    report.pop("figure", None)
    return app.response_class(json.dumps(report, indent=2),
                              mimetype="application/json")


if __name__ == "__main__":
    # Cloud Run and most container platforms inject the port to listen on.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
