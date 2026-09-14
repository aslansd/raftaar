"""LeRobot adapter tests.

The fixtures below write the LeRobotDataset v2.x layout as published on the
Hugging Face Hub -- `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`
and `data/chunk-000/episode_*.parquet` -- rather than mocking the reader. The
adapter tracks a file format, so the tests have to exercise the file format.

Two robot shapes are covered, because they are the two the adapter has to tell
apart:

* a **joint-space arm** with a named gripper, the SO-100/SO-101 convention that
  most of the hub follows
* a **Cartesian** dataset whose state columns are named `ee_x`/`ee_y`/`ee_z`

Getting the second wrong is the expensive failure: treating joint angles as a
position produces numbers that look entirely plausible and mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from raftaar import scan
from raftaar.lerobot import (
    LeRobotFormatError,
    infer_gripper_dim,
    infer_spatial_dims,
    load_lerobot,
    segment_phases,
)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
CARTESIAN_NAMES = ["ee_x", "ee_y", "ee_z", "gripper"]


def _write_lerobot(root: Path, n_episodes=40, n_frames=60, names=JOINT_NAMES,
                   seed=0, bimodal=False, codebase_version="v2.1"):
    """Write a minimal but structurally faithful LeRobotDataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    n_dims = len(names)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    for ep in range(n_episodes):
        t = np.linspace(0, 1, n_frames)
        state = np.zeros((n_frames, n_dims), dtype="float32")

        # A reach whose mid-trajectory detour goes one way or the other. With
        # `bimodal` the two groups are separated; without it they are one blob.
        side = 1.0 if (bimodal and ep % 2 == 0) else (-1.0 if bimodal else 0.0)
        target = rng.uniform(-1, 1, size=3)
        for d in range(min(3, n_dims)):
            state[:, d] = (t * target[d]
                           + side * 0.6 * np.sin(np.pi * t)
                           + rng.normal(0, 0.02, n_frames))
        for d in range(3, n_dims - 1):
            state[:, d] = rng.normal(0, 0.05, n_frames)

        # Gripper: open, closed through the middle, open again.
        gripper = np.ones(n_frames, dtype="float32")
        gripper[n_frames // 3: 2 * n_frames // 3] = 0.0
        state[:, -1] = gripper

        action = np.diff(state, axis=0, prepend=state[:1]).astype("float32")

        table = pa.table({
            "observation.state": pa.array(state.tolist()),
            "action": pa.array(action.tolist()),
            "timestamp": pa.array((np.arange(n_frames) / 30.0).tolist()),
            "frame_index": pa.array(np.arange(n_frames).tolist()),
            "episode_index": pa.array([ep] * n_frames),
            "index": pa.array((np.arange(n_frames) + ep * n_frames).tolist()),
            "task_index": pa.array([0] * n_frames),
        })
        pq.write_table(table, root / "data" / "chunk-000"
                       / f"episode_{ep:06d}.parquet")
        episodes_meta.append({"episode_index": ep, "tasks": ["pick up the cube"],
                              "length": n_frames})

    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": codebase_version,
        "repo_id": "test/fixture",
        "robot_type": "so100",
        "fps": 30,
        "total_episodes": n_episodes,
        "total_frames": n_episodes * n_frames,
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [n_dims],
                                  "names": names},
            "action": {"dtype": "float32", "shape": [n_dims], "names": names},
        },
    }, indent=2))
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes_meta))
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the cube"}))
    return root


@pytest.fixture(scope="module")
def joint_dataset(tmp_path_factory):
    return _write_lerobot(tmp_path_factory.mktemp("lerobot") / "joints")


# --------------------------------------------------------------------------
# reading the format
# --------------------------------------------------------------------------

class TestLoading:
    def test_reads_a_lerobot_directory(self, joint_dataset):
        data = load_lerobot(joint_dataset)

        assert len(data["episodes"]) == 40
        episode = data["episodes"][0]
        assert episode["observation.state"].shape == (60, 6)
        assert episode["action"].shape == (60, 6)
        assert len(episode["phase"]) == 60

    def test_does_not_import_lerobot_or_torch(self, joint_dataset):
        """The whole point of reading the format rather than the library."""
        import subprocess
        import sys

        code = (
            "import sys; from raftaar.lerobot import load_lerobot; "
            f"load_lerobot({str(joint_dataset)!r}); "
            "bad=[m for m in ('torch','lerobot','torchvision') if m in sys.modules]; "
            "assert not bad, bad; print('ok')"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_max_episodes_reads_a_prefix(self, joint_dataset):
        assert len(load_lerobot(joint_dataset, max_episodes=5)["episodes"]) == 5

    def test_metadata_is_carried_through(self, joint_dataset):
        info = load_lerobot(joint_dataset)["info"]

        assert info["dataset_name"] == "test/fixture"
        assert info["robot_type"] == "so100"
        assert info["fps"] == 30
        assert info["_adapter"]["codebase_version"] == "v2.1"
        assert "pick up the cube" in info["_adapter"]["tasks"]

    def test_a_non_lerobot_directory_says_so(self, tmp_path):
        (tmp_path / "data").mkdir()
        with pytest.raises(LeRobotFormatError) as exc:
            load_lerobot(tmp_path)
        assert "meta/info.json" in str(exc.value)

    def test_scan_runs_on_an_adapted_dataset(self, joint_dataset):
        report = scan(load_lerobot(joint_dataset))

        assert report["n_episodes"] == 40
        assert report["n_frames"] == 40 * 60
        assert isinstance(report["findings"], list)


# --------------------------------------------------------------------------
# the two inferences the adapter has to make
# --------------------------------------------------------------------------

class TestSpatialDims:
    def test_cartesian_columns_are_found_by_name(self):
        dims, how = infer_spatial_dims(CARTESIAN_NAMES, 4)
        assert dims == [0, 1, 2]
        assert how == "cartesian-by-name"

    def test_joint_space_falls_back_and_says_so(self):
        """The label matters more than the choice.

        Comparing the first three joints is defensible; calling it a position
        is not. A scan that silently treated joint angles as xyz would produce
        plausible numbers that mean something different.
        """
        dims, how = infer_spatial_dims(JOINT_NAMES, 6)
        assert dims == [0, 1, 2]
        assert how == "first-3-joints"

    def test_the_choice_reaches_the_manifest(self, joint_dataset):
        info = load_lerobot(joint_dataset)["info"]
        assert info["spatial_dims"] == [0, 1, 2]
        assert info["_adapter"]["spatial_dims_from"] == "first-3-joints"

    def test_cartesian_dataset_is_labelled_differently(self, tmp_path):
        root = _write_lerobot(tmp_path / "cart", n_episodes=20,
                              names=CARTESIAN_NAMES)
        info = load_lerobot(root)["info"]
        assert info["_adapter"]["spatial_dims_from"] == "cartesian-by-name"


class TestGripperAndPhases:
    def test_gripper_is_found_by_name(self):
        assert infer_gripper_dim(JOINT_NAMES) == 5
        assert infer_gripper_dim(CARTESIAN_NAMES) == 3

    def test_gripper_falls_back_to_the_last_channel(self):
        actions = np.zeros((10, 4))
        assert infer_gripper_dim(None, actions) == 3

    def test_phases_follow_the_gripper(self, joint_dataset):
        data = load_lerobot(joint_dataset)
        assert data["info"]["_adapter"]["phase_method"] == "gripper"

        phases = data["episodes"][0]["phase"]
        assert set(phases) == {"approach", "transport", "release"}
        # Order matters: approach precedes transport precedes release.
        first = {p: int(np.argmax(phases == p)) for p in set(phases.tolist())}
        assert first["approach"] < first["transport"] < first["release"]

    def test_without_a_gripper_it_splits_into_thirds_and_says_so(self):
        """A weak segmentation is still better than skipping the dataset --
        provided the record says it was a guess."""
        states = np.random.default_rng(0).normal(size=(60, 5))
        actions = np.diff(states, axis=0, prepend=states[:1])

        phases, method = segment_phases(actions, states, gripper_dim=None)
        assert method == "thirds"
        assert set(phases) == {"approach", "transport", "release"}

    def test_a_constant_gripper_falls_back_to_thirds(self):
        """A channel that never moves carries no boundary information."""
        states = np.zeros((60, 6))
        states[:, 5] = 1.0
        actions = np.zeros((60, 6))

        _, method = segment_phases(actions, states, gripper_dim=5)
        assert method == "thirds"

    def test_empty_episode_does_not_crash(self):
        phases, method = segment_phases(np.zeros((0, 4)), np.zeros((0, 4)), 3)
        assert len(phases) == 0
        assert method == "empty"


# --------------------------------------------------------------------------
# what the detectors do with adapted data
# --------------------------------------------------------------------------

class TestDetectionOnAdaptedData:
    def test_bimodal_strategies_are_caught_through_the_adapter(self, tmp_path):
        """End to end: parquet on disk, through the adapter, to a finding."""
        root = _write_lerobot(tmp_path / "bi", n_episodes=60, bimodal=True,
                              seed=1)
        report = scan(load_lerobot(root))
        assert "AVERAGING_HAZARD" in {f["id"] for f in report["findings"]}

    def test_unimodal_data_does_not_raise_the_hazard(self, tmp_path):
        root = _write_lerobot(tmp_path / "uni", n_episodes=60, bimodal=False,
                              seed=1)
        report = scan(load_lerobot(root))
        assert "AVERAGING_HAZARD" not in {f["id"] for f in report["findings"]}

    def test_missing_image_features_are_reported_not_assumed(self, joint_dataset):
        """No hub dataset ships precomputed image features.

        Reporting "no shards" there would claim a check that never ran.
        """
        report = scan(load_lerobot(joint_dataset))

        assert report["shards"]["available"] is False
        assert "not assessed" in report["shards"]["reason"]
        assert "REPRESENTATION_SHARD" not in {f["id"] for f in report["findings"]}


# --------------------------------------------------------------------------
# the CLI path
# --------------------------------------------------------------------------

class TestCLIWithLeRobot:
    def test_format_is_detected_from_the_directory(self, tmp_path):
        """Both formats carry meta/info.json, so the data files decide."""
        from raftaar.cli import detect_format

        lerobot = _write_lerobot(tmp_path / "lr", n_episodes=10)
        assert detect_format(lerobot) == "lerobot"

        from raftaar import DatasetSpec, FaultSpec, build_dataset
        native = build_dataset(
            DatasetSpec(name="native", n_episodes=10, seed=0, faults=FaultSpec()),
            tmp_path / "d")
        assert detect_format(Path(native)) == "raftaar"

        (tmp_path / "empty").mkdir()
        assert detect_format(tmp_path / "empty") == "unknown"

    def test_scan_runs_end_to_end_on_a_lerobot_directory(self, tmp_path):
        import json
        import subprocess

        root = _write_lerobot(tmp_path / "hub", n_episodes=30, bimodal=True,
                              seed=2)
        out = subprocess.run(
            ["raftaar", "scan", str(root), "--out", str(tmp_path / "r")],
            capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

        # The interpretation is announced, not silent.
        assert "read as LeRobotDataset" in out.stdout
        assert "phases by gripper" in out.stdout

        report = json.loads((tmp_path / "r" / "report.json").read_text())
        assert report["adapter"]["source"] == "lerobot"
        assert report["adapter"]["phase_method"] == "gripper"

        # And it reaches the human-readable report too.
        markdown = (tmp_path / "r" / "report.md").read_text()
        assert "How this dataset was read" in markdown
        assert "not assessed" in markdown

    def test_max_episodes_is_honoured_by_the_cli(self, tmp_path):
        import json
        import subprocess

        root = _write_lerobot(tmp_path / "big", n_episodes=40)
        subprocess.run(["raftaar", "scan", str(root), "--out", str(tmp_path / "r"),
                        "--max-episodes", "12"], capture_output=True, text=True)
        report = json.loads((tmp_path / "r" / "report.json").read_text())
        assert report["n_episodes"] == 12

    def test_a_directory_of_neither_format_fails_clearly(self, tmp_path):
        import subprocess

        (tmp_path / "junk" / "data").mkdir(parents=True)
        out = subprocess.run(["raftaar", "scan", str(tmp_path / "junk")],
                             capture_output=True, text=True)
        assert out.returncode != 0
        combined = out.stdout + out.stderr
        assert "neither" in combined and "--format" in combined

    def test_the_figure_renders_without_image_features(self, tmp_path):
        """The fourth panel plots image features; most datasets have none."""
        pytest.importorskip("matplotlib")
        from raftaar.report import plot

        root = _write_lerobot(tmp_path / "nofeat", n_episodes=20)
        data = load_lerobot(root)
        out = tmp_path / "fig.png"
        plot(data, scan(data), out)
        assert out.stat().st_size > 5000
