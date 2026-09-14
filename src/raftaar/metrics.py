"""
Diagnostics that run on a demonstration dataset *before* any training.

Every metric here answers a question of the form "what will a policy learn
from this, and where will that be wrong?" -- not "is this file corrupt".
That distinction is the whole product thesis: structural validators are
commoditising, distributional ones are not.

All of it is CPU-only and O(minutes) on a laptop.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

EPS = 1e-9


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _standardize(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    return (X - mu) / (sd + EPS)


def _resample(traj: np.ndarray, k: int) -> np.ndarray:
    """Resample a (T, d) trajectory to k evenly spaced waypoints."""
    if len(traj) == 0:
        return np.zeros((k, traj.shape[1] if traj.ndim > 1 else 1))
    idx = np.linspace(0, len(traj) - 1, k)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    w = (idx - lo)[:, None]
    return traj[lo] * (1 - w) + traj[hi] * w


def strategy_signature(traj: np.ndarray, k: int = 8) -> np.ndarray:
    """
    A description of *how* a phase was performed, with *where* removed.

    Resample the trajectory to k waypoints and subtract the straight line
    between its own endpoints. What remains is the detour shape -- the
    operator's strategy -- independent of where the object happened to be.
    Without this, variation in task conditions swamps variation in strategy
    and the mode detector sees nothing.
    """
    w = _resample(traj, k)
    line = np.linspace(0, 1, k)[:, None] * (w[-1] - w[0])[None, :] + w[0][None, :]
    return (w - line).ravel()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices with the same number of rows."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (den + EPS))


def participation_ratio(X: np.ndarray) -> float:
    """Effective (intrinsic) dimensionality of a point cloud."""
    C = np.cov(_standardize(X), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + EPS))


# ----------------------------------------------------------------------------
# 1. Averaging hazard  (the metric the whole tool exists for)
# ----------------------------------------------------------------------------

def averaging_hazard(episodes: list[dict], phase: str, n_waypoints: int = 8,
                     max_modes: int = 3,
                     spatial_dims: Sequence[int] | None = None) -> dict:
    """
    Do the demonstrations of this phase form more than one strategy, and if so,
    is their *average* a strategy nobody demonstrated?

    A regression policy trained with an MSE-like objective converges toward the
    conditional mean. When that mean sits in a region with no demonstration
    density, the converged policy is one no operator would recognise -- and the
    training loss will not tell you.
    """
    # Which state columns describe *where the robot is*. The synthetic robot
    # puts end-effector xyz first, but a joint-space arm does not, and silently
    # treating the first three joint angles as a position is the kind of wrong
    # that still produces plausible-looking numbers. The adapter supplies this;
    # None keeps the original behaviour.
    dims = list(range(3)) if spatial_dims is None else list(spatial_dims)
    sigs = [
        strategy_signature(
            ep["observation.state"][ep["phase"] == phase][:, dims], n_waypoints)
        for ep in episodes
    ]
    S = _standardize(np.asarray(sigs))

    # Project first. Fitting full-covariance mixtures in the raw waypoint space
    # needs far more episodes than anyone has; BIC would always return one mode.
    # We deliberately do NOT re-standardise: PCA's variance scaling is what
    # keeps the mode-separating direction dominant over noise directions.
    n_pc = int(min(3, S.shape[1], max(2, len(S) // 8)))
    S = PCA(n_pc, random_state=0).fit_transform(S)

    # How many strategies? BIC over Gaussian mixtures.
    bics, models = [], []
    reg = 1e-3 * float(S.var())          # scale-aware: stops components collapsing
    for k in range(1, min(max_modes, max(1, len(S) // 12)) + 1):
        gm = GaussianMixture(k, covariance_type="full", reg_covar=reg,
                             random_state=0, n_init=8).fit(S)
        bics.append(gm.bic(S))
        models.append(gm)
    # BIC keeps falling as k grows -- real demonstration clusters always have
    # sub-structure -- so the minimum over-splits. Take the elbow instead: keep
    # adding strategies only while each one buys a substantial fraction of what
    # the first split bought.
    bics = np.asarray(bics)
    gains = -np.diff(bics)
    n_modes = 1
    if len(gains) and gains[0] > 0:
        n_modes = 2
        for j in range(1, len(gains)):
            if gains[j] < 0.40 * gains[0]:
                break
            n_modes = j + 2
    gm = models[n_modes - 1]

    # How far is the average demonstration from the nearest real strategy,
    # measured in units of the spread *within* a strategy? A regression policy
    # converges toward that average. If this number is large, the thing it
    # converges to is not a demonstration anybody gave.
    spread = float(np.sqrt(np.mean([np.trace(c) / c.shape[0]
                                    for c in np.atleast_3d(gm.covariances_)])))
    d_to_modes = np.linalg.norm(S.mean(0)[None] - gm.means_, axis=1)
    hazard = float(d_to_modes.min() / (spread + EPS))

    # Guard against BIC splitting a single blob: real strategies separate.
    sep, explained = 0.0, 0.0
    if n_modes > 1:
        lab = gm.predict(S)
        if len(set(lab)) > 1:
            sep = float(silhouette_score(S, lab))

            # Is the split just a consequence of the task conditions? Paths to
            # objects on the left and on the right *should* differ; that is the
            # policy's job, not a hazard. A hazard is a split the conditions
            # cannot predict -- a free choice the operator made.
            C = np.asarray([ep["observation.state"][0] for ep in episodes])
            C = C[:, C.std(0) > 0.02]
            if C.shape[1] and min(np.bincount(lab)) >= 3:
                acc = cross_val_score(
                    LogisticRegression(max_iter=1000), _standardize(C), lab,
                    cv=min(5, int(min(np.bincount(lab)))), scoring="accuracy").mean()
                base = float(np.bincount(lab).max() / len(lab))
                explained = float(max(0.0, (acc - base) / (1.0 - base + EPS)))

    return {
        "phase": phase,
        "n_modes": n_modes,
        "hazard_ratio": hazard,       # in within-strategy standard deviations
        "separation": sep,            # silhouette of the strategy split
        "condition_explained": explained,  # 1.0 = the split is just the task
        "bic_curve": [float(b) for b in bics],
    }


# ----------------------------------------------------------------------------
# 2. Coverage geometry
# ----------------------------------------------------------------------------

def coverage(episodes: list[dict], slack: float = 2.1) -> dict:
    """
    Which task conditions were never demonstrated?

    Sparsity is not a hole. With n demonstrations spread over a d-dimensional
    condition space, the expected distance to the nearest demonstration is
    ~0.5 * n^(-1/d). We only call a region a gap when it is further from data
    than that -- so the detector does not fire simply because the dataset is
    small. Boundary regions are excluded via the convex hull: not demonstrating
    beyond the edge of your workspace is a choice, not a defect.
    """
    from scipy.spatial import Delaunay

    starts = np.asarray([ep["observation.state"][0] for ep in episodes])
    cond_dims = np.where(starts.std(0) > 0.02)[0]

    result = {
        "condition_dims": cond_dims.tolist(),
        "intrinsic_dim": participation_ratio(
            np.concatenate([ep["observation.state"] for ep in episodes])),
        "gap_ratio": 0.0,
        "gap_cells": [],
    }
    if not 1 <= len(cond_dims) <= 3:
        return result

    C = starts[:, cond_dims]
    lo, hi = C.min(0), C.max(0)
    Cn = (C - lo) / (hi - lo + EPS)
    n, d = Cn.shape

    g = {1: 200, 2: 40, 3: 14}[d]
    probes = np.stack(np.meshgrid(*([np.linspace(0, 1, g)] * d), indexing="ij"),
                      axis=-1).reshape(-1, d)

    if d >= 2:
        try:
            probes = probes[Delaunay(Cn).find_simplex(probes) >= 0]
        except Exception:
            pass
    if len(probes) == 0:
        return result

    dist, _ = NearestNeighbors(n_neighbors=1).fit(Cn).kneighbors(probes)
    expected = 0.5 * n ** (-1.0 / d)
    holes = dist.ravel() > slack * expected

    result["gap_ratio"] = float(holes.mean())
    result["gap_cells"] = (probes[holes][:400] * (hi - lo) + lo).tolist()
    return result


# ----------------------------------------------------------------------------
# 3. Conditional action entropy
# ----------------------------------------------------------------------------

def conditional_action_entropy(episodes: list[dict], k: int = 15,
                               max_samples: int = 3000, seed: int = 0,
                               names: list[str] | None = None) -> dict:
    """
    In near-identical states, did the operator take near-identical actions?

    H(a|s) is estimated as the local variance of actions among state-space
    neighbours, normalised by the global action variance. It is high when the
    demonstrator is inconsistent *or* genuinely multimodal -- read it together
    with the averaging hazard to tell those apart.
    """
    S = np.concatenate([ep["observation.state"] for ep in episodes])
    A = np.concatenate([ep["action"] for ep in episodes])
    P = np.concatenate([ep["phase"] for ep in episodes])
    rng = np.random.default_rng(seed)

    # Neighbours are drawn from within the same phase and normalised against
    # that phase's own action variance. Otherwise ordinary phase structure --
    # a gripper that only moves during a grasp -- reads as inconsistency.
    # Phase progress is appended to the neighbour space. Without it, the
    # ordinary within-phase sequence (descend, then close) looks like an
    # operator giving different actions in the same state.
    T = np.concatenate([np.linspace(0, 1, len(ep["action"])) for ep in episodes])

    # Neighbours are taken from *other* episodes only. Consecutive frames of the
    # same episode are trivially similar; the question worth asking is whether
    # two different demonstrations, in the same state, did the same thing.
    E = np.concatenate([np.full(len(ep["action"]), i)
                        for i, ep in enumerate(episodes)])
    names = names or [f"dim_{i}" for i in range(A.shape[1])]

    per_phase, dominant = {}, {}
    for ph in np.unique(P):
        m = P == ph
        Sp = np.column_stack([_standardize(S[m]), 3.0 * _standardize(T[m, None])])
        Ap, Ep = A[m], E[m]
        if len(Sp) < k * 4:
            continue
        sel = (rng.choice(len(Sp), max_samples, replace=False)
               if len(Sp) > max_samples else np.arange(len(Sp)))
        pool = min(4 * k, len(Sp))
        _, nbrs = NearestNeighbors(n_neighbors=pool).fit(Sp).kneighbors(Sp[sel])

        acc = []
        for i, q in enumerate(sel):
            cand = nbrs[i][Ep[nbrs[i]] != Ep[q]][:k]
            if len(cand) >= 5:
                acc.append(Ap[cand].var(axis=0))
        if not acc:
            continue
        v = np.mean(acc, axis=0)
        per_phase[str(ph)] = float(v.sum() / (Ap.var(axis=0).sum() + EPS))
        dominant[str(ph)] = names[int(np.argmax(v / (Ap.var(axis=0) + EPS)))]

    overall = float(np.mean(list(per_phase.values()))) if per_phase else 0.0
    return {"overall": overall, "per_phase": per_phase, "dominant_channel": dominant}


# ----------------------------------------------------------------------------
# 4. Representation shards
# ----------------------------------------------------------------------------

def representation_shards(episodes: list[dict]) -> dict:
    """
    Do the visual features split into groups that no one intended? A camera
    that was nudged, a lighting change, a different operator session -- the
    policy sees these as distinct worlds and silently splits its capacity.
    """
    # Real LeRobot datasets ship encoded video, not precomputed image features.
    # Returning "no shards detected" there would be a lie -- we did not look --
    # so the absence is reported as its own state and `_findings` skips it.
    if not all("observation.image_features" in ep for ep in episodes):
        return {"available": False, "n_shards": 1, "silhouette": 0.0,
                "reason": "no observation.image_features in this dataset; "
                          "visual sharding was not assessed"}

    M = np.asarray([ep["observation.image_features"].mean(0) for ep in episodes])
    Mz = _standardize(M)

    best = {"k": 1, "silhouette": 0.0, "labels": np.zeros(len(M), int)}
    for k in range(2, min(4, len(M) // 5) + 1):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Mz)
        s = silhouette_score(Mz, lab)
        if s > best["silhouette"]:
            best = {"k": k, "silhouette": float(s), "labels": lab}

    out = {"available": True, "n_shards": 1, "silhouette": best["silhouette"],
           "shard_sizes": [len(M)], "cka_between": 1.0}

    if best["silhouette"] > 0.55:
        lab = best["labels"]
        sizes = [int((lab == c).sum()) for c in range(best["k"])]
        order = np.argsort(sizes)[::-1]
        a = np.concatenate([episodes[i]["observation.image_features"]
                            for i in np.where(lab == order[0])[0]])
        b = np.concatenate([episodes[i]["observation.image_features"]
                            for i in np.where(lab == order[1])[0]])
        n = min(len(a), len(b))
        out.update(n_shards=best["k"], shard_sizes=sorted(sizes, reverse=True),
                   cka_between=linear_cka(a[:n], b[:n]))
    return out


# ----------------------------------------------------------------------------
# 5. Actuator health
# ----------------------------------------------------------------------------

def actuator_health(episodes: list[dict], names: list[str] | None = None) -> dict:
    """
    Action channels that carry no signal in a phase. A model will happily
    learn to emit ~0 for them and the loss will still fall, so nothing in the
    training curve flags a stuck joint or a mis-mapped channel.

    Important honesty note: idleness is often *by design* -- a gripper channel
    does nothing during an approach. On a single dataset this detector can only
    say "this channel is idle here, confirm that is intended". It becomes sharp
    when run against a reference dataset (see `scan(..., reference=...)`), where
    a channel that used to move and now does not is unambiguous.
    """
    A = np.concatenate([ep["action"] for ep in episodes])
    P = np.concatenate([ep["phase"] for ep in episodes])
    names = names or [f"dim_{i}" for i in range(A.shape[1])]

    idle = []
    for ph in np.unique(P):
        sd = A[P == ph].std(0)
        scale = sd.max() + EPS
        for i, s in enumerate(sd):
            if s / scale < 0.02:
                idle.append({"phase": str(ph), "dim": names[i],
                             "rel_std": float(s / scale)})
    return {"idle_channels": idle}


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def scan(data: dict, reference: dict | None = None) -> dict:
    """
    Run every diagnostic and turn the numbers into findings.

    `reference` is an earlier scan of a dataset you trusted. Supplying it turns
    ambiguous single-dataset observations into regressions.
    """
    eps = data["episodes"]
    info = data["info"]
    if not eps:
        raise ValueError(
            "no episodes in this dataset. If it is a LeRobotDataset, check that "
            "data/**/*.parquet exists and contains observation.state and action."
        )
    action_names = info["features"]["action"].get("names")
    # Recorded by whichever adapter built this dataset; see metrics.averaging_hazard.
    spatial_dims = info.get("spatial_dims")
    phases = list(dict.fromkeys(np.concatenate([ep["phase"] for ep in eps]).tolist()))

    report = {
        "dataset": info.get("dataset_name", "unknown"),
        "n_episodes": len(eps),
        "n_frames": int(sum(len(ep["action"]) for ep in eps)),
        "averaging": [averaging_hazard(eps, ph, spatial_dims=spatial_dims)
                      for ph in phases],
        "coverage": coverage(eps),
        "entropy": conditional_action_entropy(eps, names=action_names),
        "shards": representation_shards(eps),
        "adapter": info.get("_adapter"),
        "actuators": actuator_health(eps, action_names),
    }
    report["findings"] = _findings(report, reference)
    return report


def _findings(r: dict, reference: dict | None = None) -> list[dict]:
    f = []

    for a in r["averaging"]:
        if (a["n_modes"] > 1 and a["separation"] > 0.35
                and a["hazard_ratio"] > 1.0
                and a["condition_explained"] < 0.60):
            f.append({
                "id": "AVERAGING_HAZARD",
                "severity": "critical" if a["hazard_ratio"] > 2.0 else "warning",
                "where": f"phase '{a['phase']}'",
                "what": (f"{a['n_modes']} distinct strategies (separation "
                         f"{a['separation']:.2f}, only "
                         f"{a['condition_explained']:.0%} explained by task "
                         f"conditions); their average lies "
                         f"{a['hazard_ratio']:.1f} within-strategy SD from the "
                         f"nearest real strategy"),
                "so_what": ("A unimodal regressor (ACT, MSE behaviour cloning) will "
                            "converge to a trajectory no operator ever performed. "
                            "Use a multimodal policy class, or split the modes and "
                            "condition on them."),
            })

    c = r["coverage"]
    if c["gap_ratio"] > 0.10:
        f.append({
            "id": "COVERAGE_GAP",
            "severity": "critical" if c["gap_ratio"] > 0.20 else "warning",
            "where": f"task-condition dims {c['condition_dims']}",
            "what": (f"{c['gap_ratio']:.0%} of the interior of the demonstrated "
                     f"condition space has zero support"),
            "so_what": ("The policy will interpolate through regions it has never "
                        "seen and fail there at deployment, while validation loss "
                        "-- computed on the same holes -- stays flat."),
        })

    for ph, v in r["entropy"]["per_phase"].items():
        if v <= 0.30:
            continue
        ch = r["entropy"]["dominant_channel"].get(ph, "?")
        f.append({
            "id": "ACTION_INCONSISTENCY",
            "severity": "warning" if v > 0.50 else "info",
            "where": f"phase '{ph}', mostly on {ch}",
            "what": (f"different episodes in near-identical states disagree on "
                     f"{v:.0%} of this phase's action variance"),
            "so_what": ("Either the operator was inconsistent, or the state is "
                        "missing the variable that decides this. Check whether "
                        f"{ch} is a discrete decision the observation cannot see; "
                        "if so, action chunking or an explicit phase input will "
                        "help more than more data."),
        })

    s = r["shards"]
    if s["n_shards"] > 1:
        f.append({
            "id": "REPRESENTATION_SHARD",
            "severity": "warning",
            "where": "observation.image_features",
            "what": (f"{s['n_shards']} disjoint feature clusters "
                     f"(sizes {s['shard_sizes']}, CKA {s['cka_between']:.2f})"),
            "so_what": ("Part of this dataset was recorded under a different "
                        "calibration or lighting. Check session metadata before "
                        "mixing; otherwise capacity is spent modelling the split."),
        })

    idle = r["actuators"]["idle_channels"]
    if idle:
        ref_idle = ({(d["dim"], d["phase"]) for d in
                     reference["actuators"]["idle_channels"]}
                    if reference else None)
        new = ([d for d in idle if (d["dim"], d["phase"]) not in ref_idle]
               if ref_idle is not None else [])

        if new:
            f.append({
                "id": "CHANNEL_WENT_SILENT",
                "severity": "critical",
                "where": ", ".join(f"{d['dim']} in '{d['phase']}'" for d in new),
                "what": ("this channel carried signal in the reference dataset "
                         "and carries none here"),
                "so_what": ("A stuck actuator or a changed channel mapping. The "
                            "model will learn to emit ~0 and the training loss "
                            "will still fall. Stop collecting and check the "
                            "hardware."),
            })

        rest = [d for d in idle if d not in new]
        if rest:
            f.append({
                "id": "IDLE_CHANNEL",
                "severity": "info",
                "where": ", ".join(f"{d['dim']} in '{d['phase']}'" for d in rest),
                "what": "no variance in this phase",
                "so_what": ("Usually by design -- a gripper is idle during an "
                            "approach. Re-run with `--reference` against a "
                            "dataset you trust to tell design from defect."),
            })

    return f
