"""Tests for packaging, the CLI, reporting, and the optional integrations.

The theme is graceful degradation. Raftaar's core install is three
dependencies; matplotlib, Flask and daftar are all optional. Each of those
options has to be genuinely optional, which means tested with the package
absent, not just documented as optional.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from raftaar import DatasetSpec, FaultSpec, build_dataset, load_dataset, scan


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    spec = DatasetSpec(name="pkg", n_episodes=40, seed=3, faults=FaultSpec())
    return build_dataset(spec, tmp / "datasets")


class TestPublicAPI:
    def test_top_level_imports(self):
        import raftaar

        for name in ("scan", "DatasetSpec", "FaultSpec", "build_dataset",
                     "load_dataset"):
            assert hasattr(raftaar, name)
        assert raftaar.__version__

    def test_importing_raftaar_does_not_pull_matplotlib(self):
        """The core install has no plotting stack; importing must not need one."""
        code = (
            "import sys; import raftaar; "
            "assert 'matplotlib' not in sys.modules, "
            "'importing raftaar pulled matplotlib'; print('ok')"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_importing_raftaar_does_not_pull_flask_or_torch(self):
        code = (
            "import sys, raftaar; "
            "bad = [m for m in ('flask', 'torch', 'gunicorn') if m in sys.modules]; "
            "assert not bad, bad; print('ok')"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr


class TestReporting:
    def test_markdown_report_needs_no_matplotlib(self, dataset, tmp_path):
        from raftaar.report import write_markdown

        report = scan(load_dataset(dataset))
        out = tmp_path / "report.md"
        write_markdown(report, out)

        text = out.read_text()
        assert "Raftaar" in text
        assert report["dataset"] in text

    def test_plot_raises_a_named_error_without_matplotlib(self, dataset, tmp_path,
                                                          monkeypatch):
        """The message has to name the fix, not just fail."""
        import builtins

        from raftaar.report import PlottingUnavailable, plot

        real_import = builtins.__import__

        def blocked(name, *a, **kw):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("No module named 'matplotlib'", name="matplotlib")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", blocked)

        data = load_dataset(dataset)
        with pytest.raises(PlottingUnavailable) as exc:
            plot(data, scan(data), tmp_path / "x.png")
        assert "raftaar[plot]" in str(exc.value)


class TestCLI:
    def test_cli_entry_point_is_installed(self):
        out = subprocess.run(["raftaar", "--help"], capture_output=True,
                             text=True)
        assert out.returncode == 0
        for command in ("synth", "scan", "validate"):
            assert command in out.stdout

    def test_synth_then_scan_round_trip(self, tmp_path):
        synth = subprocess.run(
            ["raftaar", "synth", "cli_test", "--episodes", "30"],
            cwd=tmp_path, capture_output=True, text=True)
        assert synth.returncode == 0, synth.stderr

        scan_run = subprocess.run(
            ["raftaar", "scan", "datasets/cli_test", "--out", "out"],
            cwd=tmp_path, capture_output=True, text=True)
        assert scan_run.returncode == 0, scan_run.stderr

        report = json.loads((tmp_path / "out" / "report.json").read_text())
        assert report["n_episodes"] == 30
        assert (tmp_path / "out" / "report.md").exists()

    def test_synth_accepts_named_faults(self, tmp_path):
        out = subprocess.run(
            ["raftaar", "synth", "faulty", "--episodes", "30",
             "--faults", "bimodal_detour", "dead_actuator"],
            cwd=tmp_path, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert (tmp_path / "datasets" / "faulty").exists()


class TestProvenanceIsOptional:
    def test_provenance_module_imports_without_daftar(self):
        from raftaar import provenance

        assert isinstance(provenance.available(), bool)

    def test_track_yields_none_when_daftar_is_absent(self, monkeypatch):
        from raftaar import provenance

        monkeypatch.setattr(provenance, "available", lambda: False)
        with provenance.track("x") as run:
            assert run is None

    def test_record_scan_is_a_noop_on_none(self):
        from raftaar import provenance

        provenance.record_scan(None, {"findings": []})  # must not raise

    def test_tracked_scan_matches_plain_scan(self, dataset):
        """Drop-in: the report is identical whether or not daftar is present."""
        from raftaar.provenance import tracked_scan

        plain = scan(load_dataset(dataset))
        tracked = tracked_scan(dataset, label="test-scan")

        assert set(plain) == set(tracked)
        assert plain["n_episodes"] == tracked["n_episodes"]
        assert ([f["id"] for f in plain["findings"]]
                == [f["id"] for f in tracked["findings"]])


class TestSynthEnvironment:
    def test_dataset_round_trips_through_disk(self, dataset):
        data = load_dataset(dataset)
        assert len(data["episodes"]) == 40
        assert "info" in data
        for episode in data["episodes"][:3]:
            assert len(episode["action"]) > 0
            assert len(episode["phase"]) == len(episode["action"])

    def test_same_seed_gives_the_same_dataset(self, tmp_path):
        """A synthetic environment used as ground truth has to be deterministic."""
        specs = [
            DatasetSpec(name=f"seeded{i}", n_episodes=20, seed=99,
                        faults=FaultSpec())
            for i in range(2)
        ]
        reports = [scan(load_dataset(build_dataset(s, tmp_path / f"d{i}")))
                   for i, s in enumerate(specs)]

        assert reports[0]["n_frames"] == reports[1]["n_frames"]
        assert ([f["id"] for f in reports[0]["findings"]]
                == [f["id"] for f in reports[1]["findings"]])

    def test_different_seeds_give_different_trajectories(self, tmp_path):
        """Frame counts match by design -- episodes have a fixed horizon -- so
        the comparison has to be on the trajectories themselves."""
        import numpy as np

        a = load_dataset(build_dataset(
            DatasetSpec(name="s1", n_episodes=20, seed=1, faults=FaultSpec()),
            tmp_path / "a"))
        b = load_dataset(build_dataset(
            DatasetSpec(name="s2", n_episodes=20, seed=2, faults=FaultSpec()),
            tmp_path / "b"))

        assert a["episodes"][0]["action"].shape == b["episodes"][0]["action"].shape
        assert not np.allclose(a["episodes"][0]["action"],
                               b["episodes"][0]["action"])
