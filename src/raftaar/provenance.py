"""Optional provenance recording.

A Raftaar scan is a measurement, and `raftaar validate` is an experiment:
it trains two policies and rolls them out, so its numbers depend on seeds, on
scikit-learn's version, and on the exact dataset that went in. Six months later
"the tool said AVERAGING_HAZARD on this dataset" is only useful if you can say
*which* dataset, scanned with *which* version, under *which* seed.

This module records that, using [daftar](https://pypi.org/project/daftar/) when
it is installed. It is entirely optional: nothing here is imported unless you
ask for it, and every function degrades to a no-op when daftar is absent, so
`pip install raftaar` stays a three-dependency install.

    from raftaar.provenance import tracked_scan

    report = tracked_scan("datasets/field_data", label="field-data-audit")

Without daftar this behaves exactly like `scan(load_dataset(path))`.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .metrics import scan
from .synth import load_dataset

__all__ = ["available", "tracked_scan", "record_scan", "track"]


def available() -> bool:
    """True if daftar is importable."""
    try:
        import daftar  # noqa: F401
        return True
    except ImportError:
        return False


@contextlib.contextmanager
def track(label: str, seed: int | None = None, **params: Any):
    """`daftar.track` when daftar is present, a no-op context otherwise.

    Yields the run object, or ``None``. Callers must tolerate ``None`` — that is
    the price of the dependency being optional, and it is cheaper than making
    every user of a CPU audit tool install a provenance system.
    """
    if not available():
        yield None
        return

    import daftar

    with daftar.track(label, params=params or None, seed=seed) as run:
        yield run


def record_scan(run: Any, report: dict, dataset_path: str | Path | None = None) -> None:
    """Write a scan's findings into a daftar run. No-op if ``run`` is None."""
    if run is None:
        return

    from . import __version__

    run.log_param("raftaar.version", __version__)
    run.log_param("raftaar.dataset", report.get("dataset", "unknown"))
    run.log_param("raftaar.n_episodes", report.get("n_episodes", 0))
    run.log_param("raftaar.n_frames", report.get("n_frames", 0))

    if dataset_path is not None:
        # Content hash of the dataset, so a changed or re-collected dataset is
        # detectable. This is the field that makes an old verdict re-checkable.
        with contextlib.suppress(Exception):
            run.add_input(str(dataset_path))

    findings = report.get("findings", [])
    run.log_result("findings.count", len(findings))
    run.log_result("findings.ids", sorted(f["id"] for f in findings))

    for severity in ("critical", "warning", "info"):
        n = sum(1 for f in findings if f.get("severity") == severity)
        run.log_result(f"findings.n_{severity}", n)

    # Each detector's headline number, so two scans of the same dataset can be
    # diffed rather than eyeballed.
    for phase_result in report.get("averaging", []):
        phase = phase_result.get("phase", "?")
        for key in ("gap", "n_modes", "confounded"):
            if key in phase_result:
                run.log_result(f"averaging.{phase}.{key}", phase_result[key])

    shards = report.get("shards", {})
    if "n_shards" in shards:
        run.log_result("shards.n_shards", shards["n_shards"])


def tracked_scan(path: str | Path, label: str | None = None,
                 reference: dict | None = None, seed: int | None = None) -> dict:
    """Scan a dataset, recording provenance when daftar is available.

    Returns the same report `scan()` returns, so this is a drop-in replacement
    whether or not daftar is installed.
    """
    data = load_dataset(path)
    with track(label or f"scan:{Path(path).name}", seed=seed) as run:
        report = scan(data, reference=reference)
        record_scan(run, report, dataset_path=path)
    return report
