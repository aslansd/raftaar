"""Command line interface: raftaar synth | scan | validate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import scan
from .report import PlottingUnavailable, plot, write_markdown
from .synth import DatasetSpec, FaultSpec, build_dataset, load_dataset

SEV_MARK = {"critical": "[!!]", "warning": "[! ]", "info": "[  ]"}


def _synth(args):
    faults = FaultSpec(**{f: True for f in args.faults})
    spec = DatasetSpec(name=args.name, n_episodes=args.episodes,
                       faults=faults, seed=args.seed)
    path = build_dataset(spec, args.out)
    print(f"wrote {args.episodes} episodes -> {path}")
    if faults.active():
        print("injected faults:", ", ".join(faults.active()))


def _scan(args):
    data = load_dataset(args.path)
    report = scan(data)
    out = Path(args.out or args.path)
    out.mkdir(parents=True, exist_ok=True)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, out / "report.md")

    # The figure is a nicety; the findings are the product. Missing matplotlib
    # should not cost you the scan you just waited for.
    figure = None
    try:
        plot(data, report, out / "report.png")
        figure = out / "report.png"
    except PlottingUnavailable as exc:
        print(f"\n  (no figure: {exc})")

    print(f"\nRaftaar  {report['dataset']}  "
          f"({report['n_episodes']} episodes, {report['n_frames']:,} frames)\n")
    if not report["findings"]:
        print("  no findings.\n")
    for f in report["findings"]:
        print(f"  {SEV_MARK[f['severity']]} {f['id']}  ({f['where']})")
        print(f"       {f['what']}")
    print(f"\n  -> {out/'report.json'}\n  -> {out/'report.md'}")
    if figure is not None:
        print(f"  -> {figure}")
    print()


def _validate(args):
    from .validate import evaluate
    data = load_dataset(args.path)
    print(f"\ntraining policies on {data['info']['dataset_name']} ...\n")
    res = evaluate(data["episodes"], n_trials=args.trials)
    labels = {"nominal": "nominal conditions", "gap": "inside coverage gap"}
    for cond, label in labels.items():
        print(f"  {label}")
        for name, r in res.items():
            print(f"    {name:<26} success {r[cond]['success_rate']:6.1%}   "
                  f"collision {r[cond]['collision_rate']:6.1%}")
        print()


def main(argv=None):
    p = argparse.ArgumentParser("raftaar")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a dataset with known faults")
    s.add_argument("name")
    s.add_argument("--episodes", type=int, default=40)
    s.add_argument("--out", default="datasets")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--faults", nargs="*", default=[],
                   choices=["bimodal_detour", "coverage_hole",
                            "demonstrator_jitter", "camera_shard",
                            "dead_actuator"])
    s.set_defaults(func=_synth)

    c = sub.add_parser("scan", help="diagnose a dataset without training")
    c.add_argument("path")
    c.add_argument("--out", default=None)
    c.set_defaults(func=_scan)

    v = sub.add_parser("validate", help="train policies and test the predictions")
    v.add_argument("path")
    v.add_argument("--trials", type=int, default=60)
    v.set_defaults(func=_validate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
