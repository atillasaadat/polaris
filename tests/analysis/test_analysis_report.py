"""The shared reporting convention, and the SISO validity boundary it carries.

``analysis/common/report.py`` is the piece every future analysis tool reuses, so
its margin arithmetic and its pass/fail sense are pinned here rather than only
exercised through the control analysis. The rows that verify requirements assert
on the **structured** report; nothing here parses rendered text for a verdict.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analysis.common.report import AnalysisReport, Criterion
from analysis.control import (
    MAX_SISO_COUPLING_RATIO,
    axis_margins,
    control_analysis_report,
    siso_coupling,
)

# ── The shared primitives ────────────────────────────────────────────────────


def test_criterion_margin_runs_in_the_declared_direction():
    """A ``min`` criterion and a ``max`` criterion measure margin oppositely."""
    lower = Criterion("gm", "REQ-X-001", threshold=6.0, measured=15.0, units="dB")
    assert lower.passes
    assert lower.margin == pytest.approx(9.0)
    assert lower.margin_pct == pytest.approx(150.0)

    upper = Criterion(
        "Ms", "REQ-X-001", threshold=2.0, measured=1.02, units="-", sense="max"
    )
    assert upper.passes
    assert upper.margin == pytest.approx(0.98)
    assert upper.margin_pct == pytest.approx(49.0)

    assert not replace(lower, measured=5.9).passes
    assert not replace(upper, measured=2.1).passes


def test_a_missing_measurement_fails_rather_than_passing_silently():
    """``nan`` is not a pass. An analysis that could not measure has not passed."""
    unmeasured = Criterion(
        "pm", "REQ-X-001", threshold=30.0, measured=float("nan"), units="deg"
    )
    assert not unmeasured.passes
    assert np.isnan(unmeasured.margin)


def test_an_unknown_sense_is_refused():
    """A typo in the direction would silently invert every verdict on that row."""
    with pytest.raises(ValueError, match="sense must be one of"):
        Criterion("x", "", threshold=1.0, measured=1.0, units="-", sense="minimum")


def test_report_renders_provenance_assumptions_and_verdict(tmp_path):
    """The text rendering is self-contained: config, assumptions, table, verdict."""
    report = AnalysisReport(
        title="Example analysis",
        config_path="config/spacecraft/example.yaml",
        provenance={"gains": "Kp=1"},
        assumptions=("linear regime",),
        criteria=(
            Criterion("passing", "REQ-X-001", 1.0, 2.0, "-"),
            Criterion("failing", "REQ-X-002", 10.0, 2.0, "-"),
        ),
        warnings=("qualified by something",),
    )
    assert not report.passes
    assert [c.name for c in report.failures()] == ["failing"]
    assert [c.name for c in report.by_requirement("REQ-X-001")] == ["passing"]

    text = report.format_text()
    for expected in (
        "config/spacecraft/example.yaml",
        "linear regime",
        "REQ-X-002",
        "qualified by something",
        "FAIL",
    ):
        assert expected in text

    written = report.write_text(tmp_path / "nested" / "report.txt")
    assert written.read_text().startswith("Example analysis — FAIL")


# ── The control analysis through the shared shape ────────────────────────────


@pytest.mark.verifies("REQ-ACTL-006", "REQ-ACTL-007", "REQ-ACTL-008")
def test_control_analysis_report_passes_on_the_committed_configuration(
    vehicle, record_property, reference_config
):
    """Every criterion of all three requirements passes on the shipped config."""
    report = control_analysis_report(vehicle, reference_config)
    assert report.failures() == []
    assert report.passes

    for requirement in ("REQ-ACTL-006", "REQ-ACTL-007", "REQ-ACTL-008"):
        criteria = report.by_requirement(requirement)
        assert criteria, f"{requirement} has no criteria"
        assert all(c.passes for c in criteria)

    assert report.config_path == str(reference_config)
    assert report.assumptions
    worst = min(c.margin_pct for c in report.criteria if np.isfinite(c.margin_pct))
    record_property("margin_pct", f"{worst:.0f}")


def test_the_report_carries_the_assumptions_the_numbers_rest_on(
    vehicle, reference_config
):
    """A margin without its assumptions is not a result."""
    assumptions = " ".join(
        control_analysis_report(vehicle, reference_config).assumptions
    )
    for topic in ("Small angle", "Unsaturated", "Integrator active", "SISO"):
        assert topic in assumptions


# ── The SISO validity boundary ───────────────────────────────────────────────


def test_siso_coupling_matches_its_closed_form(vehicle):
    """ρ = ‖h‖/(J_min ω_c), with ‖h‖ the array's largest body-axis capacity."""
    crossover = 0.3
    ratio, limit = siso_coupling(vehicle, crossover)
    capacity = (
        np.max(np.sum(np.abs(vehicle.wheel_spin_axes), axis=1))
        * vehicle.wheel_max_momentum_nms
    )
    diagonal = np.min(vehicle.principal_moments_kgm2) * crossover
    assert ratio == pytest.approx(capacity / diagonal, rel=1e-12)
    assert limit == pytest.approx(MAX_SISO_COUPLING_RATIO * diagonal, rel=1e-12)
    # The limit is exactly the momentum at which the ratio reaches the bound.
    assert siso_coupling(
        replace(
            vehicle,
            wheel_max_momentum_nms=limit
            / float(np.max(np.sum(np.abs(vehicle.wheel_spin_axes), axis=1))),
        ),
        crossover,
    )[0] == pytest.approx(MAX_SISO_COUPLING_RATIO, rel=1e-12)


def test_the_reference_vehicle_warns_at_full_stored_momentum(vehicle, reference_config):
    """A momentum-biased vehicle is not a per-axis problem, and the report says so.

    This is a *warning*, not a failure: the margins are those of the loop as
    analysed, and what the warning qualifies is the claim that the analysed loop
    is the vehicle at every operating point. Asserted rather than left implicit
    because a warning nobody checks is a comment.
    """
    for i in range(3):
        result = axis_margins(vehicle, i)
        assert result.siso_coupling_ratio > MAX_SISO_COUPLING_RATIO
        assert not result.siso_assumption_holds
        # Finite, positive, and inside what the array can physically store —
        # compared against the *array* rather than a single wheel, because the
        # re-baselined vehicle's certified limit (0.51 N.m.s) is above one RW-X's
        # 0.5 N.m.s capacity while still being a small fraction of the four-wheel
        # pyramid's. Which of the two it sits between is a property of the
        # vehicle; that it is a real bound below the hardware is the claim.
        array_capacity = (
            vehicle.wheel_spin_axes.shape[1] * vehicle.wheel_max_momentum_nms
        )
        assert 0.0 < result.siso_momentum_limit_nms < array_capacity

    report = control_analysis_report(vehicle, reference_config)
    assert len(report.warnings) == 3
    assert all("MIMO" in w for w in report.warnings)
    # Warnings qualify a report; they do not fail it.
    assert report.passes


def test_a_low_momentum_array_raises_no_siso_warning(vehicle, reference_config):
    """The warning must be able to *not* fire, or it says nothing when it does."""
    quiet = replace(vehicle, wheel_max_momentum_nms=1.0e-4)
    for i in range(3):
        assert axis_margins(quiet, i).siso_assumption_holds
    assert control_analysis_report(quiet, reference_config).warnings == ()
