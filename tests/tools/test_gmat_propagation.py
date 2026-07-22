"""Tests for the GMAT propagation golden harness (design doc §23.1, REQ-VV-002).

GMAT cannot run in CI, so these exercise the pieces that must be correct without
it: script generation, turning a ``ReportFile`` into SI samples, fixture assembly,
and the drift comparison. The committed fixture is checked for schema conformance
and for agreeing with the committed ``.script`` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gmat import (
    build_propagation_fixture,
    compare_propagation,
    regenerate_propagation_fixture,
    samples_from_rows,
    write_propagation_script,
)
from gmat import propagation
from gmat.propagation import KM_TO_M, PROPAGATION_CASES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "golden" / "gmat_propagation.json"
_SCRIPT_DIR = _REPO_ROOT / "tools" / "gmat" / "scripts"

# Two rows of a circular equatorial orbit; t=0 at the epoch, then GMAT's slightly
# overshot "600 s" sample.
_SAMPLE_REPORT = """\
sat.ElapsedSecs sat.EarthMJ2000Eq.X sat.EarthMJ2000Eq.Y sat.EarthMJ2000Eq.Z \
sat.EarthMJ2000Eq.VX sat.EarthMJ2000Eq.VY sat.EarthMJ2000Eq.VZ
0 6878.137 0 0 0 7.612608173223869 0
600.0000003841706 5416.465947577085 4239.182141581899 0 -4.691787 5.994028 0
"""

_CASE = PROPAGATION_CASES[0]


@pytest.mark.verifies("REQ-VV-002")
def test_samples_convert_km_to_si_and_keep_gmat_elapsed_time():
    samples = samples_from_rows(propagation.parse_report(_SAMPLE_REPORT))
    assert len(samples) == 2
    assert samples[0]["position_m"] == [6878137.0, 0.0, 0.0]
    # The overshoot is carried verbatim — the C++ side propagates to exactly this.
    assert samples[1]["t_s"] == 600.0000003841706
    assert samples[1]["position_m"][1] == pytest.approx(4239182.141581899)
    assert samples[1]["velocity_m_s"][0] == pytest.approx(-4691.787)


@pytest.mark.verifies("REQ-VV-002")
def test_samples_reject_missing_state_column():
    with pytest.raises(ValueError, match="missing columns"):
        samples_from_rows([{"ElapsedSecs": 0.0, "X": 1.0}])


@pytest.mark.verifies("REQ-VV-002")
def test_samples_require_the_epoch_sample():
    rows = propagation.parse_report(_SAMPLE_REPORT)[1:]
    with pytest.raises(ValueError, match="expected the epoch"):
        samples_from_rows(rows)


@pytest.mark.verifies("REQ-VV-002")
def test_samples_reject_empty_report():
    with pytest.raises(ValueError, match="no state rows"):
        samples_from_rows([])


@pytest.mark.verifies("REQ-VV-002")
def test_write_script_emits_force_model_and_sampling_loop():
    script = write_propagation_script(_CASE, "/tmp/rf.txt")
    assert "Earth.Mu = 398600.4418;" in script  # matches wgs84::kGM exactly
    assert "fm.PointMasses = {Earth};" in script
    assert "prop.Type = RungeKutta89;" in script
    assert "rf.Filename = '/tmp/rf.txt';" in script
    assert "While sat.ElapsedSecs < 5400" in script
    assert "Propagate prop(sat) {sat.ElapsedSecs = 600};" in script
    # Default precision (10 digits) would quantise position to ~1 m.
    assert "rf.Precision = 16;" in script


@pytest.mark.verifies("REQ-VV-002")
def test_third_body_script_selects_the_de424_ephemeris():
    case = next(c for c in PROPAGATION_CASES if c["name"] == "third_body")
    script = write_propagation_script(case, "/tmp/rf.txt")
    # Verified against GMAT R2026a: this loads DE424AllPlanets.bsp; without it GMAT
    # silently stays on its DE405 default.
    assert "SolarSystem.EphemerisSource = 'DE424';" in script
    assert "fm.PointMasses = {Earth, Sun, Luna};" in script


@pytest.mark.verifies("REQ-VV-002")
def test_write_script_is_ascii_only():
    # GMAT R2026a rejects any non-ASCII byte in a script file (em-dashes, etc.).
    for case in PROPAGATION_CASES:
        assert write_propagation_script(case, "rf.txt").isascii()


@pytest.mark.verifies("REQ-VV-002")
def test_write_script_rejects_quote_injection():
    with pytest.raises(ValueError, match="single quote"):
        write_propagation_script(_CASE, "/tmp/rf.txt'; Stop; %")


@pytest.mark.verifies("REQ-VV-002")
def test_build_fixture_matches_loader_schema():
    samples = samples_from_rows(propagation.parse_report(_SAMPLE_REPORT))
    fixture = build_propagation_fixture(
        (_CASE,), {"two_body": samples}, generated_utc="2026-07-21"
    )
    assert fixture["schema_version"] == "1.0"
    assert fixture["case"] == "gmat_propagation"
    assert fixture["category"] == "force_model"
    assert fixture["provenance"]["gmat_version"] == "R2026a"
    case = fixture["cases"][0]
    assert case["name"] == "two_body"
    assert case["gmat_script"] == "tools/gmat/scripts/prop_two_body.script"
    assert case["initial_state"]["position_m"] == [6878137.0, 0.0, 0.0]
    assert case["initial_state"]["attitude_quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert case["environment"]["gravity_degree"] == 0
    assert case["spacecraft"]["mass_kg"] == 12.0
    assert case["samples"] == samples


@pytest.mark.verifies("REQ-VV-002")
def test_build_fixture_rejects_case_without_samples():
    with pytest.raises(ValueError, match="no samples for cases"):
        build_propagation_fixture((_CASE,), {})


@pytest.mark.verifies("REQ-VV-002")
def test_regenerate_runs_gmat_once_per_case(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        script = Path(cmd[-1])
        report = script.parent / script.name.replace(".script", "_report.txt")
        report.write_text(_SAMPLE_REPORT)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(propagation.subprocess, "run", fake_run)
    fixture = regenerate_propagation_fixture("GmatConsole", str(tmp_path))
    assert [c["name"] for c in fixture["cases"]] == [
        c["name"] for c in PROPAGATION_CASES
    ]


@pytest.mark.verifies("REQ-VV-002")
def test_regenerate_raises_when_gmat_writes_no_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        propagation.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="no report"):
        regenerate_propagation_fixture("GmatConsole", str(tmp_path))


@pytest.mark.verifies("REQ-VV-002")
def test_compare_passes_on_agreement_and_flags_drift():
    committed = json.loads(_FIXTURE.read_text())
    assert compare_propagation(committed, committed) == []

    drifted = json.loads(_FIXTURE.read_text())
    case = drifted["cases"][0]
    case["samples"][3]["position_m"][0] += 10.0
    messages = compare_propagation(committed, drifted)
    assert messages and "position_m" in messages[0]

    # A shift below the tolerance is not drift.
    nudged = json.loads(_FIXTURE.read_text())
    nudged["cases"][0]["samples"][3]["position_m"][0] += 0.01
    assert compare_propagation(committed, nudged) == []


@pytest.mark.verifies("REQ-VV-002")
def test_compare_flags_missing_case_and_shifted_sample_times():
    committed = json.loads(_FIXTURE.read_text())

    dropped = json.loads(_FIXTURE.read_text())
    del dropped["cases"][0]
    assert any("missing" in m for m in compare_propagation(committed, dropped))

    shifted = json.loads(_FIXTURE.read_text())
    shifted["cases"][0]["samples"][2]["t_s"] += 1.0
    assert any("sampled t=" in m for m in compare_propagation(committed, shifted))

    truncated = json.loads(_FIXTURE.read_text())
    truncated["cases"][0]["samples"].pop()
    assert any("samples" in m for m in compare_propagation(committed, truncated))


@pytest.mark.verifies("REQ-VV-002")
def test_committed_fixture_is_self_consistent():
    fixture = json.loads(_FIXTURE.read_text())
    assert [c["name"] for c in fixture["cases"]] == [
        c["name"] for c in PROPAGATION_CASES
    ]
    for case, spec in zip(fixture["cases"], PROPAGATION_CASES):
        assert case["initial_state"]["position_m"] == [
            v * KM_TO_M for v in spec["position_km"]
        ]
        assert case["samples"][0]["t_s"] == 0.0
        assert case["samples"][0]["position_m"] == case["initial_state"]["position_m"]
        # Sample times must be strictly increasing and land on the nominal grid to
        # within GMAT's sub-microsecond stop-condition overshoot.
        times = [s["t_s"] for s in case["samples"]]
        assert times == sorted(times) and len(set(times)) == len(times)
        for i, t in enumerate(times):
            assert abs(t - i * spec["sample_step_s"]) < 1.0e-3
        assert case["position_tolerance_m"] > 0.0
        assert case["tolerance_rationale"]


@pytest.mark.verifies("REQ-VV-002")
def test_committed_scripts_match_the_generator():
    # The committed .script files are the provenance record for the fixture, so a
    # generator change that is not re-run would leave them lying.
    for case in PROPAGATION_CASES:
        committed = (_SCRIPT_DIR / f"prop_{case['name']}.script").read_text()
        body = write_propagation_script(case, "PLACEHOLDER")
        # Only the report path differs (it is absolute and machine-specific).
        strip = lambda text: [  # noqa: E731
            ln for ln in text.splitlines() if not ln.startswith("rf.Filename")
        ]
        assert strip(committed) == strip(body)
