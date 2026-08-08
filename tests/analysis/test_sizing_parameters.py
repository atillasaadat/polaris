"""Derived flight tuning, its justification, and the CLI gate (design doc §8.5, §12).

The parameter derivations are cross-checked against :mod:`analysis.control` —
the momentum envelope through the same ``siso_coupling`` the margin work uses,
the loop crossover through the same ``axis_margins`` — so the two packages
cannot describe different vehicles from one config.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.control.margins import axis_margins
from analysis.control.vehicle import load_vehicle
from analysis.control.plant import siso_coupling
from analysis.sizing.parameters import (
    MAX_BANDWIDTH_FRACTION,
    implied_bandwidth,
)
from analysis.sizing.plots import write_all
from analysis.sizing.report import (
    format_budget,
    format_derived,
    sizing_analysis,
    sizing_report,
)


def _criterion(report, fragment):
    matches = [c for c in report.criteria if fragment in c.name]
    assert len(matches) == 1, f"{fragment!r} matched {[c.name for c in matches]}"
    return matches[0]


def _derived(analysis, name):
    return next(p for p in analysis.derived if p.name == name)


def test_the_committed_gains_recover_their_own_bandwidth_and_damping(vehicle):
    """Inverting ``Kp = J wn²`` and ``Kd = 2 ζ wn J`` returns the design point.

    Asserted as a **round trip against the committed gains**, not against a
    transcribed (wn, ζ). The design point moves whenever the vehicle is retuned —
    it moved by 4.5x in the 50 kg re-baseline — and a literal here would only
    ever record which retune was most recent. What must hold at every design
    point is that the inversion is exact: recovering (wn, ζ) and pushing them
    back through the forward formulas must return the gains the vehicle flies.
    That is what makes the report's derivation rows a consistency check rather
    than a restatement.
    """
    wn, zeta = implied_bandwidth(vehicle)
    j_mean = float(np.mean(vehicle.principal_moments_kgm2))
    assert wn > 0.0 and 0.0 < zeta < 2.0
    assert j_mean * wn**2 == pytest.approx(vehicle.pid.kp_nm_per_rad, rel=1e-9)
    assert 2.0 * zeta * wn * j_mean == pytest.approx(
        vehicle.pid.kd_nm_per_radps, rel=1e-9
    )


def test_the_derived_gains_reproduce_the_committed_ones(vehicle):
    """Round trip: derive Kp and Kd from the implied (wn, zeta) and get them back."""
    analysis = sizing_analysis(vehicle)
    for name, committed in (
        ("PidKpNmPerRad", vehicle.pid.kp_nm_per_rad),
        ("PidKdNmPerRadps", vehicle.pid.kd_nm_per_radps),
    ):
        parameter = _derived(analysis, name)
        assert parameter.derived == pytest.approx(committed, rel=1e-9)
        assert parameter.ratio == pytest.approx(1.0, rel=1e-9)


def test_the_momentum_envelope_is_derived_through_the_control_package(vehicle):
    """Same function, same crossover, same bound — not a second derivation.

    ``analysis.control.plant.siso_coupling`` is the one implementation of the
    SISO validity boundary, and ``tests/analysis/test_control_momentum_envelope.py``
    already guards the committed config against it. This asserts that the sizing
    package reuses it rather than re-deriving a number free to drift.
    """
    crossover = axis_margins(vehicle, 0).gain_crossover_rad_s
    _, bound = siso_coupling(vehicle, crossover)
    parameter = _derived(sizing_analysis(vehicle), "MomentumEnvelopeNms")
    # The claim is *identity with the one implementation*, to 1e-12 — that is
    # what "reuses it rather than re-deriving it" means and it is the whole
    # point of the test. The two lines that used to follow pinned the bound and
    # the committed value at their 12 kg magnitudes; both moved by ~190x in the
    # re-baseline, and neither was testing this package. The committed value is
    # guarded against this bound by
    # tests/analysis/test_control_momentum_envelope.py, which is where that
    # check belongs.
    assert parameter.derived == pytest.approx(bound, rel=1e-12)
    assert parameter.committed <= bound


def test_the_bandwidth_criterion_is_written_on_the_sampling_bound(
    vehicle, reference_config
):
    """wn must sit below ws/10, and the criterion fails when it does not.

    The failing side is a vehicle with a 100x stiffer proportional gain — a real
    way to break this, and one the sampled-data margins in
    :mod:`analysis.control` would also object to.
    """
    report = sizing_report(vehicle, reference_config)
    bandwidth = _criterion(report, "closed-loop bandwidth")
    assert bandwidth.passes
    assert bandwidth.threshold == pytest.approx(
        MAX_BANDWIDTH_FRACTION * 2.0 * np.pi / vehicle.control_period_s
    )

    # The failing case is derived, not a literal: "100 N·m/rad" was 100x the
    # committed gain on the 12 kg bus and is merely a brisk loop on the 50 kg
    # one, so a fixed number silently stops testing the failure it was written
    # for. Place the gain to put wn at twice the sampling bound instead, which
    # is a violation on any vehicle.
    j_mean = float(np.mean(vehicle.principal_moments_kgm2))
    bound = MAX_BANDWIDTH_FRACTION * 2.0 * np.pi / vehicle.control_period_s
    stiff = dataclasses.replace(
        vehicle,
        pid=dataclasses.replace(vehicle.pid, kp_nm_per_rad=j_mean * (2.0 * bound) ** 2),
    )
    assert not _criterion(
        sizing_report(stiff, reference_config), "closed-loop bandwidth"
    ).passes


def test_the_damping_criterion_catches_an_underdamped_pair(vehicle, reference_config):
    """zeta comes from Kp and Kd together; an underdamped design fails it."""
    assert _criterion(sizing_report(vehicle, reference_config), "damping ratio").passes
    twitchy = dataclasses.replace(
        vehicle, pid=dataclasses.replace(vehicle.pid, kd_nm_per_radps=5.0e-3)
    )
    assert not _criterion(
        sizing_report(twitchy, reference_config), "damping ratio"
    ).passes


def test_the_desaturation_ordering_invariant_is_enforced_both_ways(
    vehicle, reference_config
):
    """exit < enter < envelope. Break either inequality and the report says so."""
    report = sizing_report(vehicle, reference_config)
    assert _criterion(report, "desaturation entry inside the envelope").passes
    assert _criterion(report, "desaturation exit below entry").passes

    # Break each inequality relative to the vehicle's own envelope, so the
    # case stays a violation whatever the envelope currently is.
    envelope = vehicle.momentum_envelope_nms
    late = dataclasses.replace(vehicle, momentum_desat_enter_nms=1.5 * envelope)
    assert not _criterion(
        sizing_report(late, reference_config), "desaturation entry inside the envelope"
    ).passes

    no_hysteresis = dataclasses.replace(
        vehicle, momentum_desat_exit_nms=1.5 * vehicle.momentum_desat_enter_nms
    )
    assert not _criterion(
        sizing_report(no_hysteresis, reference_config), "desaturation exit below entry"
    ).passes


def test_the_bdot_gain_clears_the_avanzini_floor(vehicle, reference_config):
    """``k >= 2 omega_o (1 + sin xi) J_min`` at the worst-case xi.

    The reference gain is 9x the floor, which the config's own comment states.
    A gain below the floor does not converge, and that is the side asserted.
    """
    report = sizing_report(vehicle, reference_config)
    gain = _criterion(report, "BdotGainNms above the Avanzini")
    assert gain.passes
    # J_min comes from the vehicle, never a transcribed constant: this test
    # was written against a 12 kg bus and a literal here would have to be
    # re-typed at every retune, which is how a test stops checking the
    # formula and starts checking a memory of one run.
    j_min = float(np.min(vehicle.principal_moments_kgm2))
    floor = 4.0 * vehicle.orbit.mean_motion_rad_s * j_min
    assert gain.threshold == pytest.approx(floor)
    assert vehicle.bdot_gain_nms >= floor

    feeble = dataclasses.replace(vehicle, bdot_gain_nms=1.0e-4)
    assert not _criterion(
        sizing_report(feeble, reference_config), "BdotGainNms above the Avanzini"
    ).passes


def test_the_detumble_exit_recommendation_refuses_to_go_below_the_noise_floor(vehicle):
    """When the two bounds disagree the LARGER governs, and the text says which.

    The recommendation is the **larger** of the two bounds, and the reasoning
    text says which one governed and why — that refusal is the point of the row.
    """
    analysis = sizing_analysis(vehicle)
    parameter = _derived(analysis, "DetumbleExitRadps")
    floor = analysis.mtq.noise_floor.rate_worst_radps
    momentum_bound = (
        0.5
        * analysis.wheels.usable_momentum_nms
        / float(np.max(vehicle.principal_moments_kgm2))
    )
    # The RULE, not which bound happens to win today. Both are real lower
    # bounds — the wheels must be able to take the handover, and B-dot cannot
    # certify a rate it cannot measure — so the recommendation is the larger.
    # Which one governs is a property of the design and it flipped in the 50 kg
    # re-baseline: on the 12 kg bus the noise floor won and the two were
    # *incompatible*; with the larger certified envelope the momentum bound
    # wins and the design closes. A test naming the winner would have failed on
    # the fix.
    assert parameter.derived == pytest.approx(max(floor, momentum_bound))
    assert parameter.derived >= floor
    assert "governs" in parameter.reasoning or "refused" in parameter.reasoning


def test_a_quiet_magnetometer_lets_the_momentum_bound_govern_instead(vehicle):
    """With the floor out of the way the wheel-envelope bound sets the threshold."""
    quiet = dataclasses.replace(vehicle, mag_noise_t=1.0e-10)
    analysis = sizing_analysis(quiet)
    parameter = _derived(analysis, "DetumbleExitRadps")
    assert parameter.derived == pytest.approx(
        0.5
        * analysis.wheels.usable_momentum_nms
        / float(np.max(vehicle.principal_moments_kgm2))
    )
    assert "governs" in parameter.reasoning


def test_every_derived_parameter_carries_its_justification(vehicle):
    """Formula, inputs and reasoning on all of them — the package's differentiator."""
    for parameter in sizing_analysis(vehicle).derived:
        assert parameter.formula
        assert parameter.inputs
        assert len(parameter.reasoning) > 80, parameter.name
        assert np.isfinite(parameter.derived)


def test_every_figure_and_the_text_report_are_written(
    vehicle, reference_config, tmp_path
):
    """Four figures plus the rendered report, into a caller-supplied directory.

    A smoke suite: nothing asserts on pixels, but a plotting call that raises
    would take the toolkit down at the moment someone needed a picture. The
    report file carries the criteria table, the budget and the justifications,
    so it stands alone as the record.
    """
    paths = write_all(vehicle, tmp_path, reference_config)
    assert len(paths) == 5
    for path in paths:
        assert path.parent == tmp_path
        assert path.stat().st_size > 0
    assert [p.suffix for p in paths] == [".png"] * 4 + [".txt"]
    text = paths[-1].read_text()
    assert str(reference_config) in text
    assert "Disturbance-torque budget" in text
    assert "Derived flight parameters" in text


def test_the_renderings_are_self_contained(vehicle):
    """The budget and derivation blocks name their formulas and their inputs."""
    analysis = sizing_analysis(vehicle)
    budget_text = format_budget(analysis)
    assert "3*mu/(2*R^3)" in budget_text
    assert "summed, not RSS" in budget_text
    derived_text = format_derived(analysis)
    assert "Kp = J * wn^2" in derived_text
    assert "avanzini" in derived_text.lower() or "Avanzini" in derived_text


def test_the_cli_gates_on_the_reference_vehicle(reference_config, tmp_path):
    """The exit status is the report's verdict, whatever that verdict currently is.

    Deliberately **not** "the reference vehicle fails". It did when this tool was
    written, and the tool is what got it fixed; a test asserting the committed
    design is broken becomes a liability the moment someone repairs it, and would
    have to be edited by whoever does — which is a test punishing the outcome it
    exists to encourage. What must hold forever is that the process exit status
    and the structured report agree, because CI reads the former and reviewers
    read the latter.
    """
    from analysis.sizing.__main__ import main

    gate = sizing_report(load_vehicle(reference_config), reference_config)
    expected = 0 if gate.passes else 1
    assert (
        main([str(reference_config), "--out", str(tmp_path), "--no-plots"]) == expected
    )


def test_the_cli_gates_non_zero_on_a_design_that_cannot_pass(
    reference_config, tmp_path
):
    """A FAIL still has to reach the shell — pinned with a vehicle that must fail.

    The companion to the test above: since the reference design now passes, the
    non-zero path needs its own case or nothing covers it. A tip-off far beyond
    what the array can hold is a failure by construction and independent of any
    future retune of the committed vehicle.
    """
    from analysis.sizing.__main__ import main

    status = main(
        [
            str(reference_config),
            "--out",
            str(tmp_path),
            "--no-plots",
            "--tipoff-deg-s",
            "500",
        ]
    )
    assert status == 1


def test_the_cli_accepts_the_assumption_overrides(reference_config, tmp_path):
    """Tip-off, desaturation interval and slew rate are settable from the shell.

    Reusability is the point: the same tool has to size a different vehicle and a
    different concept of operations without an edit to the package.
    """
    from analysis.sizing.__main__ import main

    assert (
        main(
            [
                str(reference_config),
                "--out",
                str(tmp_path),
                "--no-plots",
                "--tipoff-deg-s",
                "0.1",
                "--desat-orbits",
                "0.5",
                "--slew-deg-s",
                "0.05",
            ]
        )
        in (0, 1)  # the overrides must be accepted; the verdict is the design's
    )
