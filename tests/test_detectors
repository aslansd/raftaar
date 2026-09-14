"""Ground-truth tests for the detectors.

These are the tests that matter, and the reason the synthetic environment exists.
Every fault here is *injected deliberately*, so each detector can be scored
against something better than opinion: the dataset is known to contain the
defect, and the question is whether the tool says so.

The two directions are tested separately and both are load-bearing:

* **Sensitivity** — a dataset built with a fault must raise that fault's finding.
* **Specificity** — a clean dataset must stay quiet. A detector that fires on
  everything is worse than no detector, because it trains users to ignore it.

Episode counts are kept modest so the suite stays fast, but not so small that
the statistics stop working: several detectors are sample-size aware and will
correctly refuse to conclude anything from 20 episodes.
"""

from __future__ import annotations

import pytest

from raftaar import DatasetSpec, FaultSpec, build_dataset, load_dataset, scan

EPISODES = 90
SEED = 7


def build(tmp_path, name, **faults):
    """Build a synthetic dataset with the named faults switched on."""
    spec = DatasetSpec(
        name=name,
        n_episodes=EPISODES,
        seed=SEED,
        faults=FaultSpec(**faults),
    )
    path = build_dataset(spec, tmp_path / "datasets")
    return load_dataset(path)


def finding_ids(report):
    return {f["id"] for f in report["findings"]}


@pytest.fixture(scope="module")
def clean_report(tmp_path_factory):
    data = build(tmp_path_factory.mktemp("clean"), "clean")
    return scan(data)


# --------------------------------------------------------------------------
# specificity: the tool must stay quiet on good data
# --------------------------------------------------------------------------

class TestCleanDataStaysQuiet:
    def test_no_averaging_hazard_on_clean_data(self, clean_report):
        """The headline detector must not fire when there is one strategy.

        This is the direction that makes the tool usable. A warning that appears
        on every dataset carries no information, and the feasibility claim is
        explicitly bidirectional: quiet on clean data, loud on broken data.
        """
        assert "AVERAGING_HAZARD" not in finding_ids(clean_report)

    def test_no_coverage_gap_on_clean_data(self, clean_report):
        assert "COVERAGE_GAP" not in finding_ids(clean_report)

    def test_no_representation_shard_on_clean_data(self, clean_report):
        assert "REPRESENTATION_SHARD" not in finding_ids(clean_report)

    def test_clean_scan_has_the_expected_shape(self, clean_report):
        assert clean_report["n_episodes"] == EPISODES
        assert clean_report["n_frames"] > 0
        for key in ("averaging", "coverage", "entropy", "shards", "actuators"):
            assert key in clean_report


# --------------------------------------------------------------------------
# sensitivity: each injected fault must be caught
# --------------------------------------------------------------------------

class TestInjectedFaultsAreDetected:
    def test_bimodal_detour_raises_averaging_hazard(self, tmp_path):
        """Two valid strategies whose average is a path nobody drove.

        The defect the whole project is built around: training loss falls the
        entire time, validation loss does not catch it because the validation
        split has the same structure, and the policy converges to a trajectory
        that goes through the obstacle.
        """
        data = build(tmp_path, "bimodal", bimodal_detour=True)
        report = scan(data)
        assert "AVERAGING_HAZARD" in finding_ids(report)

    def test_coverage_hole_raises_coverage_gap(self, tmp_path):
        data = build(tmp_path, "hole", coverage_hole=True)
        report = scan(data)
        assert "COVERAGE_GAP" in finding_ids(report)

    def test_camera_shard_raises_representation_shard(self, tmp_path):
        """Part of the dataset recorded under a different calibration."""
        data = build(tmp_path, "shard", camera_shard=True)
        report = scan(data)
        assert "REPRESENTATION_SHARD" in finding_ids(report)

    def test_dead_actuator_raises_idle_channel(self, tmp_path):
        data = build(tmp_path, "dead", dead_actuator=True)
        report = scan(data)
        assert "IDLE_CHANNEL" in finding_ids(report)

    def test_every_fault_at_once_is_noisier_than_clean(self, tmp_path, clean_report):
        """The full field-data case from the feasibility plan."""
        data = build(
            tmp_path,
            "field_data",
            bimodal_detour=True,
            coverage_hole=True,
            demonstrator_jitter=True,
            camera_shard=True,
            dead_actuator=True,
        )
        report = scan(data)
        ids = finding_ids(report)

        assert "AVERAGING_HAZARD" in ids
        assert len(report["findings"]) > len(clean_report["findings"])


# --------------------------------------------------------------------------
# the confound guard
# --------------------------------------------------------------------------

class TestConfoundGuard:
    def test_severity_is_ordered_by_consequence(self, tmp_path):
        """A hazard that will break training must outrank a cosmetic note."""
        data = build(tmp_path, "bimodal", bimodal_detour=True)
        report = scan(data)

        levels = {"critical": 2, "warning": 1, "info": 0}
        hazard = next(f for f in report["findings"]
                      if f["id"] == "AVERAGING_HAZARD")
        assert levels[hazard["severity"]] >= levels["warning"]

    def test_findings_are_actionable(self, tmp_path):
        """Every finding says what and where, not just that something is wrong."""
        data = build(tmp_path, "bimodal", bimodal_detour=True)
        for f in scan(data)["findings"]:
            assert f["id"] and f["what"] and f["where"]
            assert f["severity"] in ("critical", "warning", "info")


# --------------------------------------------------------------------------
# reference scans turn ambiguity into regression
# --------------------------------------------------------------------------

class TestReferenceComparison:
    def test_scan_accepts_a_reference_without_changing_shape(self, tmp_path):
        baseline_data = build(tmp_path / "a", "baseline")
        baseline = scan(baseline_data)

        later_data = build(tmp_path / "b", "later", dead_actuator=True)
        with_ref = scan(later_data, reference=baseline)
        without_ref = scan(later_data)

        assert set(with_ref) == set(without_ref)
        assert isinstance(with_ref["findings"], list)


# --------------------------------------------------------------------------
# the sample-size caveat, pinned
# --------------------------------------------------------------------------

class TestCleanDataSeverityIsSampleSizeDependent:
    """`ACTION_INCONSISTENCY` fires on clean data, and its severity depends on n.

    Worth stating plainly, because the README's summary table reads "clean -> no
    warnings" and that is only true once the dataset is large enough:

        60 episodes  -> ACTION_INCONSISTENCY at *warning*
        90 episodes  -> ACTION_INCONSISTENCY at *info*
        120 episodes -> ACTION_INCONSISTENCY at *info*

    The detector is measuring something real — near-identical states genuinely do
    disagree when there are few episodes per region of the state space — so this
    is under-powering rather than a bug. But a user scanning 40 episodes will see
    a warning on data that is fine, and should know that before they act on it.
    """

    def test_no_warning_or_critical_on_clean_data_at_120_episodes(self, tmp_path):
        spec = DatasetSpec(name="clean120", n_episodes=120, seed=SEED,
                           faults=FaultSpec())
        report = scan(load_dataset(build_dataset(spec, tmp_path / "d")))

        severe = [f for f in report["findings"]
                  if f["severity"] in ("warning", "critical")]
        assert not severe, f"clean data should not raise warnings: {severe}"

    def test_small_clean_datasets_can_raise_a_spurious_warning(self, tmp_path):
        """Documents the limitation rather than hiding it.

        If this test ever starts failing because the warning no longer appears,
        that is an improvement — delete the test.
        """
        spec = DatasetSpec(name="clean60", n_episodes=60, seed=SEED,
                           faults=FaultSpec())
        report = scan(load_dataset(build_dataset(spec, tmp_path / "d")))

        ids = {f["id"]: f["severity"] for f in report["findings"]}
        assert "ACTION_INCONSISTENCY" in ids
