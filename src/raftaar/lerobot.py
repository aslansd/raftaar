"""Read LeRobotDataset directories.

This reads the **format on disk** rather than importing `lerobot`. The library
depends on torch, torchvision and opencv; a CPU dataset audit should not pull a
GPU stack to open a parquet file. The trade is that this module tracks the
format rather than the API, and the format is the more stable of the two.

Both published layouts are read:

    v2.x   data/chunk-000/episode_000000.parquet   one episode per file
    v3.0   data/chunk-000/file-0000.parquet        many episodes per file

Episodes are separated on the `episode_index` column, which exists in both, so
the reader does not depend on the file naming. That matters: v3 names its files
`file-*.parquet`, and a reader matching `episode_*.parquet` finds nothing and
reports the dataset as empty -- or worse, matches and treats forty concatenated
episodes as one trajectory, which yields a scan rather than an error.

    meta/info.json           features, fps, codebase version
    meta/tasks.jsonl         task index -> natural-language task string
    meta/episodes.jsonl      v2.x only; v3 uses chunked parquet under
                             meta/episodes/, which this reader does not need

Two things real datasets do not have, which the detectors need:

**Phase labels.** The synthetic environment ships them; nothing on the Hub does.
`segment_phases` derives them from the gripper channel, which is the honest
generic choice for manipulation: the moments the gripper closes and opens are
the boundaries an operator actually works to. When no gripper channel can be
identified the episode is split into equal thirds instead, and the dataset
records which method was used, because the two are not equally trustworthy and
a reader should not have to guess.

**A notion of which state columns are spatial.** `averaging_hazard` compares
*where the arm went*, so it needs position-like columns. For a joint-space arm
the first three columns are joint angles, not a position -- still a legitimate
space to compare trajectories in, but not the same thing, and the difference
belongs in the record rather than in a docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "load_lerobot",
    "segment_phases",
    "infer_gripper_dim",
    "infer_spatial_dims",
    "LeRobotFormatError",
]

#: Substrings that identify a gripper channel in LeRobot feature names. These
#: cover the SO-100/101 and Aloha conventions, which between them account for
#: most of the public hub.
GRIPPER_HINTS = ("gripper", "grip", "jaw", "hand", "claw", "finger")

#: Substrings that identify Cartesian position channels, when a dataset exposes
#: them. Most teleoperated arms record joint angles instead.
SPATIAL_HINTS = ("_x", "_y", "_z", "pos_x", "pos_y", "pos_z", "ee_", "tcp_")


class LeRobotFormatError(RuntimeError):
    """The directory is not a LeRobotDataset, or is a version we cannot read."""


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _feature_names(info: dict, key: str) -> list[str] | None:
    feature = (info.get("features") or {}).get(key) or {}
    names = feature.get("names")
    # LeRobot writes names either as a flat list or as {"motors": [...]}.
    if isinstance(names, dict):
        for value in names.values():
            if isinstance(value, list):
                return [str(v) for v in value]
        return None
    if isinstance(names, list):
        # Occasionally a nested single-element list.
        if len(names) == 1 and isinstance(names[0], list):
            return [str(v) for v in names[0]]
        return [str(v) for v in names]
    return None


def _parquet_files(root: Path, info: dict) -> list[Path]:
    """Every parquet under data/, whatever it is called.

    v2.x names them `episode_000000.parquet`, v3.0 names them
    `file-0000.parquet`. Matching on the `episode_*` prefix silently found
    nothing on v3 and reported the dataset as empty, so the glob is deliberately
    permissive and the episode split is done on the `episode_index` column
    instead -- which is present in both formats.
    """
    return sorted(root.glob("data/**/*.parquet"))


def _load_table(path: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LeRobotFormatError(
            "reading LeRobot datasets needs pyarrow. Install it with: "
            "pip install 'raftaar[lerobot]'"
        ) from exc

    table = pq.read_table(path)
    out: dict[str, np.ndarray] = {}
    for name in table.column_names:
        column = table.column(name).to_pylist()
        if not column:
            continue
        first = column[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            out[name] = np.asarray(column, dtype="float32")
        elif isinstance(first, (int, float, bool, np.number)):
            out[name] = np.asarray(column, dtype="float32")
        # Strings and nested structs are metadata; the detectors do not use them.
    return out


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

def infer_gripper_dim(names: Sequence[str] | None,
                      actions: np.ndarray | None = None) -> int | None:
    """Index of the gripper channel, by name if possible, else by shape.

    Falling back to "the last channel" is a real convention -- SO-100, SO-101
    and Aloha all put the gripper last -- but it is a convention rather than a
    guarantee, so it is only used when the names give nothing.
    """
    if names:
        for i, name in enumerate(names):
            if any(hint in name.lower() for hint in GRIPPER_HINTS):
                return i
        # Names were published and none of them is a gripper. That is an answer,
        # not a reason to guess: PushT is a 2-D pusher with no gripper at all,
        # and the last-channel fallback picked its y coordinate, segmented
        # "phases by gripper" on it, and reported the result as authoritative.
        return None

    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 4:
        # No names at all. Last channel is the SO-100/SO-101/Aloha convention,
        # but only for arms: a 2- or 3-dimensional action space is not an arm,
        # so there is nothing for the convention to be true about.
        return actions.shape[1] - 1
    return None


def infer_spatial_dims(names: Sequence[str] | None,
                       n_dims: int) -> tuple[list[int], str]:
    """Which state columns to compare trajectories in.

    Returns the indices and a one-word description of how they were chosen, so
    the choice ends up in the manifest rather than being implicit.
    """
    if names:
        cartesian = [i for i, name in enumerate(names)
                     if any(hint in name.lower() for hint in SPATIAL_HINTS)]
        if 2 <= len(cartesian) <= 3:
            return cartesian, "cartesian-by-name"

    dims = list(range(min(3, n_dims)))
    if names:
        # Joint space. Comparing the first three joints is defensible -- on a
        # typical arm they carry most of the gross spatial variation -- but it
        # is not a position, and the label says so.
        return dims, f"first-{len(dims)}-joints"

    # No feature names published, so we do not know what these columns are.
    # Calling them joints would be a guess dressed as a fact.
    return dims, f"first-{len(dims)}-state-dims (names unavailable)"


def segment_phases(actions: np.ndarray, states: np.ndarray,
                   gripper_dim: int | None) -> tuple[np.ndarray, str]:
    """Derive per-frame phase labels for one episode.

    Manipulation episodes have a natural structure: move to the object, close
    the gripper, move somewhere else, open it. The gripper channel marks those
    boundaries, so when it is identifiable the segmentation follows it.

    Without a gripper the episode is cut into equal thirds. That is a weak
    segmentation and is labelled as such; it keeps the detectors running on
    datasets they would otherwise skip, and the caller can see it was a guess.
    """
    n = len(actions)
    if n == 0:
        return np.array([], dtype="<U9"), "empty"

    if gripper_dim is not None and gripper_dim < states.shape[1]:
        g = np.asarray(states[:, gripper_dim], dtype="float64")
        span = float(g.max() - g.min())
        if span > 1e-6:
            closed = g < (g.min() + 0.5 * span)
            if closed.any() and not closed.all():
                first_closed = int(np.argmax(closed))
                last_closed = n - 1 - int(np.argmax(closed[::-1]))
                phases = np.empty(n, dtype="<U9")
                phases[:first_closed] = "approach"
                phases[first_closed:last_closed + 1] = "transport"
                phases[last_closed + 1:] = "release"
                # A phase with almost no frames destabilises the mixture fits
                # downstream; fold it into its neighbour rather than pretend.
                if min((phases == p).sum() for p in set(phases.tolist())) >= 4:
                    return phases, "gripper"

    third = max(1, n // 3)
    phases = np.empty(n, dtype="<U9")
    phases[:third] = "approach"
    phases[third:2 * third] = "transport"
    phases[2 * third:] = "release"
    return phases, "thirds"


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------

def load_lerobot(path: str | Path, max_episodes: int | None = None,
                 state_key: str = "observation.state",
                 action_key: str = "action") -> dict:
    """Load a LeRobotDataset directory into the structure Raftaar scans.

    ::

        from raftaar import scan
        from raftaar.lerobot import load_lerobot

        report = scan(load_lerobot("~/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"))

    `max_episodes` reads a prefix of the dataset, which is the difference
    between a thirty-second look and a ten-minute one on the larger hub
    datasets.
    """
    root = Path(path).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise LeRobotFormatError(
            f"{root} does not look like a LeRobotDataset: no meta/info.json. "
            f"Point this at the dataset root, not at data/ or a parquet file."
        )

    info = json.loads(info_path.read_text())
    episode_meta = {e.get("episode_index"): e
                    for e in _read_jsonl(root / "meta" / "episodes.jsonl")}
    tasks = {t.get("task_index"): t.get("task")
             for t in _read_jsonl(root / "meta" / "tasks.jsonl")}

    files = _parquet_files(root, info)
    if not files:
        raise LeRobotFormatError(f"no episode parquet files under {root/'data'}")
    # NB: max_episodes counts episodes, not files -- under v3 one file holds
    # many -- so the limit is applied while reading, not by slicing `files`.

    state_names = _feature_names(info, state_key)
    action_names = _feature_names(info, action_key)

    episodes: list[dict] = []
    methods: list[str] = []
    skipped: list[str] = []

    for file in files:
        table = _load_table(file)
        if state_key not in table or action_key not in table:
            skipped.append(f"{file.name}: missing {state_key} or {action_key}")
            continue

        # v2.x wrote one episode per parquet; v3.0 concatenates many episodes
        # into `file-0000.parquet` to keep file counts down. Splitting on the
        # `episode_index` column handles both, and is the difference between
        # reading a dataset and silently gluing forty episodes into one --
        # which would still produce a scan, just a meaningless one.
        all_states = np.atleast_2d(np.asarray(table[state_key], dtype="float32"))
        all_actions = np.atleast_2d(np.asarray(table[action_key], dtype="float32"))

        if "episode_index" in table:
            idx = np.asarray(table["episode_index"]).ravel()
            # Preserve file order rather than sorting: episodes are written in
            # order and the boundaries are what matter, not the labels.
            boundaries = np.flatnonzero(np.diff(idx)) + 1
            groups = np.split(np.arange(len(idx)), boundaries)
        else:
            groups = [np.arange(len(all_states))]

        for rows in groups:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            states = all_states[rows]
            actions = all_actions[rows]
            if states.shape[0] < 8:
                skipped.append(f"{file.name}: episode with {states.shape[0]} frames")
                continue

            gripper = infer_gripper_dim(state_names, states)
            phases, method = segment_phases(actions, states, gripper)
            methods.append(method)

            episode: dict[str, Any] = {
                "observation.state": states,
                "action": actions,
                "phase": phases,
            }
            # Carry any precomputed visual features through; most hub datasets
            # have none, and representation_shards reports that rather than
            # guessing.
            for key, value in table.items():
                if "image" in key and value.ndim == 2 and value.shape[1] > 1:
                    episode["observation.image_features"] = value[rows]
                    break
            episodes.append(episode)

        if max_episodes is not None and len(episodes) >= max_episodes:
            break

    if not episodes:
        raise LeRobotFormatError(
            f"no usable episodes in {root}. Skipped: {skipped[:3]}")

    spatial_dims, spatial_how = infer_spatial_dims(state_names,
                                                   episodes[0]["observation.state"].shape[1])

    out_info = {
        "dataset_name": info.get("repo_id") or root.name,
        "robot_type": info.get("robot_type", "unknown"),
        "fps": info.get("fps"),
        "total_episodes": len(episodes),
        "features": {
            "observation.state": {"shape": [episodes[0]["observation.state"].shape[1]],
                                  "names": state_names},
            "action": {"shape": [episodes[0]["action"].shape[1]],
                       "names": action_names},
        },
        # Everything below is how this dataset was interpreted, not what it
        # contains. It belongs in the record: two scans that segmented phases
        # differently are not comparable, and nothing else would say so.
        "spatial_dims": spatial_dims,
        "_adapter": {
            "source": "lerobot",
            "codebase_version": info.get("codebase_version"),
            "phase_method": max(set(methods), key=methods.count) if methods else "none",
            "phase_method_counts": {m: methods.count(m) for m in set(methods)},
            "spatial_dims_from": spatial_how,
            "gripper_dim": infer_gripper_dim(state_names,
                                             episodes[0]["observation.state"]),
            "episodes_read": len(episodes),
            "episodes_skipped": len(skipped),
            "tasks": sorted({t for t in tasks.values() if t})[:5],
        },
    }
    if skipped:
        out_info["_adapter"]["skipped_examples"] = skipped[:5]

    return {"info": out_info, "episodes": episodes}
