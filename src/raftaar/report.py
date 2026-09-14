"""Turn a scan into something a human can act on: a markdown report and a figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .metrics import _resample, _standardize, strategy_signature
from .synth import HOLE_Y_RANGE, POST_RADIUS, POST_XY

SEV = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}


def write_markdown(report: dict, path: Path) -> None:
    L = []
    L.append(f"# Raftaar report — `{report['dataset']}`\n")
    L.append(f"{report['n_episodes']} episodes · {report['n_frames']:,} frames · "
             f"no training required\n")

    findings = report["findings"]
    if not findings:
        L.append("\n## No findings\n\nNothing in this dataset trips a detector. "
                 "Train with confidence.\n")
    else:
        n_crit = sum(f["severity"] == "critical" for f in findings)
        L.append(f"\n## {len(findings)} findings ({n_crit} critical)\n")
        for f in findings:
            L.append(f"\n### [{SEV[f['severity']]}] {f['id']} — {f['where']}\n")
            L.append(f"**Observed.** {f['what']}\n")
            L.append(f"**Consequence.** {f['so_what']}\n")

    L.append("\n## Raw measurements\n")
    L.append("\n| phase | strategies | averaging hazard | H(a|s) |")
    L.append("|---|---|---|---|")
    ent = report["entropy"]["per_phase"]
    for a in report["averaging"]:
        L.append(f"| {a['phase']} | {a['n_modes']} | {a['hazard_ratio']:.2f}x | "
                 f"{ent.get(a['phase'], float('nan')):.3f} |")

    c = report["coverage"]
    L.append(f"\n- Condition-space gap ratio: **{c['gap_ratio']:.1%}**")
    L.append(f"- Intrinsic dimensionality of visited states: "
             f"**{c['intrinsic_dim']:.2f}**")
    L.append(f"- Feature shards: **{report['shards']['n_shards']}** "
             f"(CKA between largest two: {report['shards']['cka_between']:.2f})")
    path.write_text("\n".join(L) + "\n")


class PlottingUnavailable(ImportError):
    """matplotlib is an optional dependency; the markdown report does not need it."""


def _pyplot():
    """Import matplotlib on demand.

    Keeping it out of module scope means `raftaar scan` produces its markdown
    report on a machine with only numpy, scipy and scikit-learn installed. A
    dataset audit should not require a plotting stack to run.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PlottingUnavailable(
            "matplotlib is needed for figures. Install it with: "
            "pip install 'raftaar[plot]'"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot(data: dict, report: dict, path: Path) -> None:
    """Write the four-panel diagnostic figure. Requires the `plot` extra."""
    plt = _pyplot()
    eps = data["episodes"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"Raftaar — {report['dataset']}", fontsize=13, y=0.98)

    # -- 1. the averaging hazard, drawn ------------------------------------
    a0 = ax[0, 0]
    sigs = np.asarray([
        strategy_signature(e["observation.state"][e["phase"] == "approach"][:, :3])
        for e in eps])
    k = max(1, report["averaging"][0]["n_modes"])
    lab = (KMeans(k, n_init=10, random_state=0).fit_predict(_standardize(sigs))
           if k > 1 else np.zeros(len(eps), int))

    colors = ["#3b7dd8", "#2ba05a", "#b07aa1", "#8c8c8c"]
    for i, e in enumerate(eps):
        t = e["observation.state"][e["phase"] == "approach"]
        a0.plot(t[:, 0], t[:, 1], color=colors[lab[i] % 4], lw=0.8, alpha=0.5)
    mean_traj = np.stack([_resample(
        e["observation.state"][e["phase"] == "approach"][:, :3], 60) for e in eps]).mean(0)
    a0.plot(mean_traj[:, 0], mean_traj[:, 1], color="#d1342f", lw=2.6,
            label="mean of demonstrations")
    a0.add_patch(plt.Circle(POST_XY, POST_RADIUS, color="#d1342f", alpha=0.22))
    a0.text(POST_XY[0], POST_XY[1], "obstacle", ha="center", va="center", fontsize=8)
    a0.set_title(f"Approach strategies: {k} mode(s)\n"
                 f"averaging hazard {report['averaging'][0]['hazard_ratio']:.1f}x",
                 fontsize=10)
    a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)"); a0.legend(fontsize=8)

    # -- 2. coverage --------------------------------------------------------
    a1 = ax[0, 1]
    starts = np.asarray([e["observation.state"][0] for e in eps])
    cd = report["coverage"]["condition_dims"]
    if len(cd) >= 2:
        a1.scatter(starts[:, cd[0]], starts[:, cd[1]], s=26, color="#3b7dd8",
                   edgecolor="white", linewidth=0.5)
        for g in report["coverage"]["gap_cells"]:
            a1.scatter(g[0], g[1], s=60, marker="s", facecolor="none",
                       edgecolor="#d1342f", linewidth=0.9)
    a1.axhspan(*HOLE_Y_RANGE, color="#d1342f", alpha=0.07)
    a1.set_title(f"Task-condition coverage — gap "
                 f"{report['coverage']['gap_ratio']:.0%}", fontsize=10)
    a1.set_xlabel("object x (m)"); a1.set_ylabel("object y (m)")

    # -- 3. conditional action entropy -------------------------------------
    a2 = ax[1, 0]
    ph = list(report["entropy"]["per_phase"])
    val = [report["entropy"]["per_phase"][p] for p in ph]
    bars = a2.bar(ph, val, color=["#d1342f" if v > 0.50 else ("#e8a33d" if v > 0.30 else "#3b7dd8") for v in val])
    a2.axhline(0.30, ls="--", lw=1, color="#666")
    a2.set_title("Conditional action entropy H(a|s) by phase", fontsize=10)
    a2.set_ylabel("local action var / global var")
    for b, v in zip(bars, val):
        a2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=8)

    # -- 4. feature shards ---------------------------------------------------
    a3 = ax[1, 1]
    M = np.asarray([e["observation.image_features"].mean(0) for e in eps])
    P = PCA(2).fit_transform(_standardize(M))
    ns = report["shards"]["n_shards"]
    slab = (KMeans(ns, n_init=10, random_state=0).fit_predict(_standardize(M))
            if ns > 1 else np.zeros(len(M), int))
    for c in range(max(ns, 1)):
        m = slab == c
        a3.scatter(P[m, 0], P[m, 1], s=30, color=colors[c % 4],
                   edgecolor="white", linewidth=0.5, label=f"shard {c} (n={m.sum()})")
    a3.set_title(f"Episode feature space — {ns} shard(s), "
                 f"CKA {report['shards']['cka_between']:.2f}", fontsize=10)
    a3.set_xlabel("PC1"); a3.set_ylabel("PC2")
    if ns > 1:
        a3.legend(fontsize=8)

    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
