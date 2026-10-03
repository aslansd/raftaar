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
    meta/episodes.jsonl      v2.x per-episode metadata
    meta/episodes/           v3 per-episode metadata, chunked parquet. This is
                             where a dataset's own labels live -- DROID puts
                             `collector_id` here, not in the data files -- so it
                             is read whenever grouping keys are wanted.

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


def _hashable(value: Any) -> Any:
    """Collapse a grouping value to something a partition can be built from.

    `tasks` arrives as a list of strings per episode, which is unhashable, so
    `set(groups)` fails and the grouping silently cannot be used. Joining is
    the right collapse rather than taking the first element: two episodes
    labelled with the same *set* of tasks belong together, and an episode with
    two tasks is not the same group as one with the first of them.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return " | ".join(str(v) for v in value)
    return value


def _read_episode_metadata(root: Path) -> dict[int, dict]:
    """Per-episode metadata, from either layout.

    v2.x writes `meta/episodes.jsonl`; v3.0 writes chunked parquet under
    `meta/episodes/`. Columns beyond the bookkeeping ones are the dataset's own
    labels -- who collected an episode, which building, which task variant --
    and they are the only free ground truth on the hub for whether a detected
    strategy split is real. They are not in the data files, so a reader that
    only looks there finds nothing and reports the column as absent.
    """
    out: dict[int, dict] = {}

    for record in _read_jsonl(root / "meta" / "episodes.jsonl"):
        index = record.get("episode_index")
        if index is not None:
            out[int(index)] = dict(record)

    episodes_dir = root / "meta" / "episodes"
    if episodes_dir.is_dir():
        for path in sorted(episodes_dir.glob("**/*.parquet")):
            table = safe_read = None
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(path)
            except Exception:
                continue
            columns = {name: table.column(name).to_pylist()
                       for name in table.column_names}
            indices = columns.get("episode_index")
            if indices is None:
                continue
            for row, index in enumerate(indices):
                if index is None:
                    continue
                record = out.setdefault(int(index), {})
                for name, values in columns.items():
                    if row < len(values) and values[row] is not None:
                        record.setdefault(name, values[row])
    return out


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
        first = next((v for v in column if v is not None), None)
        if first is None:
            continue
        if isinstance(first, (list, tuple, np.ndarray)):
            out[name] = np.asarray(column, dtype="float32")
        elif isinstance(first, (bool, int, float, np.number)):
            out[name] = np.asarray(column, dtype="float32")
        elif isinstance(first, str):
            # Strings were dropped here on the grounds that "the detectors do
            # not use them". True of the detectors, false of everything else:
            # `collector_id`, `building` and `task_category` are strings, and
            # they are the only labels on the hub that can act as ground truth
            # for a detected strategy split. Dropping them made every
            # string-valued grouping key report as absent -- which reads as
            # "this dataset has no operator labels" when it has them.
            out[name] = np.asarray(column, dtype=object)
        # Nested structs are still skipped: no current use, and they do not
        # flatten to anything a grouping key could be.
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

#: Key names other projects use for the same two columns. The hub is not
#: consistent about this: `robomme` ships `state` / `actions` / `image` with no
#: `observation.` prefix and a plural `actions`, and a reader matching only the
#: canonical names treats it as a dataset with no action column at all. Across
#: 180 datasets there are 129 distinct feature key names and only four are
#: universal, so aliases are the rule rather than the exception.
STATE_ALIASES = ("observation.state", "state", "observation.states", "obs.state")
ACTION_ALIASES = ("action", "actions", "observation.action", "act")


#: Columns that identify *who or what produced* an episode rather than what the
#: robot did. If present they are carried through per episode, because they are
#: the only labels on the hub that can act as ground truth for a detected
#: strategy split.
GROUPING_HINTS = (
    # `tasks` is first because it is the most useful: `meta/episodes/*.parquet`
    # carries which task each episode belongs to, it is semantically bound to
    # separate trajectories, and it is independent of recording conditions. It
    # is a list of strings per episode rather than a scalar.
    "tasks", "task",
    "collector_id", "operator_id", "user_id", "demonstrator",
    "session_id", "task_index", "task_category", "building",
)


def load_lerobot(path: str | Path, max_episodes: int | None = None,
                 state_key: str = "observation.state",
                 action_key: str = "action",
                 grouping_keys: Sequence[str] | None = None,
                 field_audit: Any = None,
                 dataset_id: str | None = None,
                 spread: bool = False) -> dict:
    """Load a LeRobotDataset directory into the structure Raftaar scans.

    ::

        from raftaar import scan
        from raftaar.lerobot import load_lerobot

        report = scan(load_lerobot("~/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"))

    `max_episodes` reads a prefix of the dataset, which is the difference
    between a thirty-second look and a ten-minute one on the larger hub
    datasets.

    `field_audit` is an optional :class:`raftaar.field_audit.FieldAudit`. Where
    it resolves the gripper channel for this dataset, that answer is used in
    preference to matching feature names against :data:`GRIPPER_HINTS` -- it is
    someone else's checked work and covers datasets whose `names` are
    placeholders. Where it has no answer, inference proceeds as before.

    `spread` changes what `max_episodes` means. By default it reads a **prefix**
    -- the first N episodes, which on a large dataset come from the first one or
    two files. Hub datasets are written in collection order, so a prefix is
    often a single session by a single operator. That is fine for a smoke test
    and wrong for anything that compares against a grouping: the grouping
    collapses to one level and no comparison is possible.

    With `spread=True` the same budget is drawn evenly across every file
    instead, which is what you want whenever the question involves who or what
    produced the episodes. It may return slightly fewer episodes than asked, in
    exchange for every file contributing.

    `grouping_keys` names per-episode metadata columns to carry through, for
    example ``["collector_id"]``. Defaults to whichever of
    :data:`GROUPING_HINTS` the dataset actually has. These are not used by any
    detector; they exist so a detected strategy split can be checked against a
    partition somebody else recorded.
    """
    root = Path(path).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise LeRobotFormatError(
            f"{root} does not look like a LeRobotDataset: no meta/info.json. "
            f"Point this at the dataset root, not at data/ or a parquet file."
        )

    info = json.loads(info_path.read_text())
    episode_meta = _read_episode_metadata(root)
    tasks = {t.get("task_index"): t.get("task")
             for t in _read_jsonl(root / "meta" / "tasks.jsonl")}

    files = _parquet_files(root, info)
    if not files:
        raise LeRobotFormatError(f"no episode parquet files under {root/'data'}")
    # NB: max_episodes counts episodes, not files -- under v3 one file holds
    # many -- so the limit is applied while reading, not by slicing `files`.

    state_names = _feature_names(info, state_key)
    action_names = _feature_names(info, action_key)

    # Ask the audit first; fall back to inference per episode below.
    audited_gripper = None
    gripper_source = "inferred"
    if field_audit is not None:
        key = dataset_id or info.get("repo_id") or root.name
        audited_gripper = field_audit.gripper_index(key, "state")
        if audited_gripper is not None:
            gripper_source = f"field-audit ({key})"

    probe = _load_table(files[0])
    requested = list(grouping_keys) if grouping_keys is not None else list(GROUPING_HINTS)
    # A key can live in the data table (constant per episode) or in the
    # per-episode metadata. DROID's `collector_id` is the second kind.
    meta_keys = set()
    for record in episode_meta.values():
        meta_keys.update(record)
    wanted_groups = [k for k in requested if k in probe]
    wanted_meta_groups = [k for k in requested
                          if k not in probe and k in meta_keys]
    del probe

    episodes: list[dict] = []
    methods: list[str] = []
    skipped: list[str] = []

    # With `spread`, take an equal share from every file rather than filling the
    # budget from the first one.
    per_file = None
    if spread and max_episodes is not None and files:
        # Floor, not ceiling. Rounding up exhausts the budget before the last
        # files are reached, which silently drops the tail of the dataset --
        # the opposite of what `spread` is for. Returning slightly fewer
        # episodes than asked is the right trade when the point is coverage.
        per_file = max(1, max_episodes // len(files))

    for file in files:
        taken_here = 0
        table = _load_table(file)

        # Resolve the two columns by alias, once, from the first file that has
        # them. A dataset that uses a non-canonical name is readable; one that
        # has neither is not, and the difference should be visible.
        if state_key not in table:
            state_key = next((k for k in STATE_ALIASES if k in table), state_key)
        if action_key not in table:
            action_key = next((k for k in ACTION_ALIASES if k in table), action_key)

        if state_key not in table or action_key not in table:
            present = sorted(k for k in table if not k.startswith("index"))[:8]
            skipped.append(
                f"{file.name}: no state/action column. Tried {STATE_ALIASES} "
                f"and {ACTION_ALIASES}; the file has {present}"
            )
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
            if per_file is not None and taken_here >= per_file:
                break
            states = all_states[rows]
            actions = all_actions[rows]
            if states.shape[0] < 8:
                skipped.append(f"{file.name}: episode with {states.shape[0]} frames")
                continue

            # Prefer the published audit; fall back to name matching.
            gripper = audited_gripper
            if gripper is None:
                gripper = infer_gripper_dim(state_names, states)
            phases, method = segment_phases(actions, states, gripper)
            methods.append(method)

            episode: dict[str, Any] = {
                "observation.state": states,
                "action": actions,
                "phase": phases,
            }
            # One value per episode: these columns are constant within an
            # episode by construction, so the first row is the episode's value.
            for key in wanted_groups:
                column = table.get(key)
                if column is not None and len(column):
                    episode.setdefault("meta", {})[key] = _hashable(column[rows[0]])

            # Keys that live in meta/episodes/ rather than the data files.
            if wanted_meta_groups:
                index = int(idx[rows[0]]) if "episode_index" in table else len(episodes)
                record = episode_meta.get(index, {})
                for key in wanted_meta_groups:
                    if key in record:
                        episode.setdefault("meta", {})[key] = _hashable(record[key])
            # Carry any precomputed visual features through; most hub datasets
            # have none, and representation_shards reports that rather than
            # guessing.
            for key, value in table.items():
                if "image" in key and value.ndim == 2 and value.shape[1] > 1:
                    episode["observation.image_features"] = value[rows]
                    break
            episodes.append(episode)
            taken_here += 1

        if max_episodes is not None and len(episodes) >= max_episodes:
            break

    if not episodes:
        raise LeRobotFormatError(
            f"no usable episodes in {root}.\n"
            f"  {skipped[0] if skipped else 'no files were read'}\n"
            f"  If this dataset names its columns differently again, pass "
            f"state_key= and action_key= explicitly."
        )

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
            "grouping_keys": sorted(set(wanted_groups) | set(wanted_meta_groups)),
            "grouping_keys_from_episode_meta": wanted_meta_groups,
            "sampling": ("spread across files" if spread else "prefix"),
            "n_files_available": len(files),
            "gripper_source": gripper_source,
            "codebase_version": info.get("codebase_version"),
            "phase_method": max(set(methods), key=methods.count) if methods else "none",
            "phase_method_counts": {m: methods.count(m) for m in set(methods)},
            "spatial_dims_from": spatial_how,
            "state_key": state_key,
            "action_key": action_key,
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
