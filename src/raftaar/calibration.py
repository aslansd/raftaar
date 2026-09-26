"""Check a detected strategy split against a partition somebody else recorded.

This exists because of a problem Raftaar cannot solve on its own. When the
detector is quiet on a real dataset, two explanations are indistinguishable from
the outside:

* the demonstrations really are unimodal, or
* the thresholds were tuned on a synthetic generator and do not transfer.

Hand-labelling trajectories would settle it, and is expensive. But some datasets
carry a partition recorded for other reasons — which operator collected an
episode, which session, which task variant — and those labels are free ground
truth for a *weaker but still decisive* question:

    when the data is known to contain several producers, does the detector
    recover that structure at better than chance?

A detector that cannot recover a partition it should be able to see is
mis-calibrated. One that recovers it is measuring something real, and its
silence elsewhere can then be believed.

This is not a claim that operators always demonstrate differently — most of the
time they do not, and agreement near zero is an ordinary result rather than a
failure. It is a floor: a method that never aligns with any known grouping,
across many datasets, is not detecting strategies at all.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["episode_groups", "agreement", "calibration_report"]


def episode_groups(data: dict, key: str) -> list[Any] | None:
    """Per-episode values of a metadata column, or None if it is absent."""
    episodes = data.get("episodes") or []
    values = [ep.get("meta", {}).get(key) for ep in episodes]
    if any(v is None for v in values):
        return None
    return values


def agreement(labels: Sequence[int], groups: Sequence[Any]) -> dict:
    """Compare a detected partition against a known one.

    Returns the adjusted Rand index and adjusted mutual information, both
    corrected for chance, so a value near zero means "no better than random"
    rather than "somewhat similar". Also returns the sizes of both partitions,
    because a comparison against a grouping with one level, or with as many
    levels as episodes, is meaningless and should be visible as such.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    labels = np.asarray(list(labels))
    codes = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    truth = np.asarray([codes[g] for g in groups])

    n_detected = len(set(labels.tolist()))
    n_groups = len(codes)
    usable = n_detected > 1 and 1 < n_groups < len(truth)

    return {
        "n_episodes": int(len(truth)),
        "n_detected_modes": int(n_detected),
        "n_known_groups": int(n_groups),
        "usable": bool(usable),
        "adjusted_rand": float(adjusted_rand_score(truth, labels)) if usable else None,
        "adjusted_mutual_info": (
            float(adjusted_mutual_info_score(truth, labels)) if usable else None
        ),
        # Why a comparison could not be made, so a null result is not read as a
        # negative one.
        "reason": (
            None if usable
            else "detector found a single strategy" if n_detected <= 1
            else "grouping has one level" if n_groups <= 1
            else "grouping is unique per episode"
        ),
    }


def calibration_report(data: dict, report: dict, key: str) -> dict:
    """Compare every phase's detected strategies against ``key``.

    ::

        from raftaar import scan
        from raftaar.lerobot import load_lerobot
        from raftaar.calibration import calibration_report

        data = load_lerobot("droid_1.0.1", max_episodes=200)
        print(calibration_report(data, scan(data), "collector_id"))
    """
    groups = episode_groups(data, key)
    if groups is None:
        return {"key": key, "available": False,
                "reason": f"no per-episode {key!r} in this dataset"}

    per_phase = {}
    for phase_result in report.get("averaging", []):
        labels = phase_result.get("mode_labels")
        if labels is None or len(labels) != len(groups):
            continue
        per_phase[phase_result["phase"]] = agreement(labels, groups)

    scored = [v["adjusted_rand"] for v in per_phase.values()
              if v.get("adjusted_rand") is not None]
    return {
        "key": key,
        "available": True,
        "n_known_groups": len(set(groups)),
        "per_phase": per_phase,
        # The headline: the strongest alignment found in any phase. A strategy
        # split that follows the operator will show up in the phase where the
        # operators actually differ, not necessarily in all of them.
        "best_adjusted_rand": max(scored) if scored else None,
    }
