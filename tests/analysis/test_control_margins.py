"""Stability margins of the shipped pointing loop (REQ-ACTL-006).

Two jobs, in this order. First **validate the extractor**: Polaris owns its
margin arithmetic — there is no control-systems library behind it — so the
machinery is checked against cases with closed-form answers before it is
pointed at the vehicle. Only then the headline case: the gains in
``config/spacecraft/leo_smallsat.yaml`` are read, the **sampled** 10 Hz loop is
built from them, and every axis is asserted against 6 dB / 30°.

Thresholds are the requirement values, imported from the module that declares
them — never re-typed here and never tuned to the measurement.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analysis.control import (
    MAX_SENSITIVITY_PEAK,
    MIN_GAIN_MARGIN_DB,
    MIN_PHASE_MARGIN_DEG,
    PREFERRED_PHASE_MARGIN_DEG,
    GridWindowError,
    Loop,
    axis_margins,
    control_analysis_report,
    discrete_open_loop,
    disk_margin,
    loop_margins,
    margin_report,
    open_loop,
)
from analysis.control.vehicle import PidGains

# ── The extractor, validated against closed forms ────────────────────────────


def test_extractor_reproduces_a_textbook_third_order_loop():
    """:math:`L(s)=1/[s(s+1)(s+2)]`: gain margin exactly 6, at :math:`\\omega=\\sqrt2`.

    The standard frequency-response example [ogata2010, Ch. 7]. The phase
    crosses −180° where the imaginary part of the denominator vanishes,
    :math:`\\omega=\\sqrt2`, and :math:`|L|=1/6` there. The phase margin follows
    from the unity-gain crossing, the positive root of :math:`u(u+1)(u+4)=1` in
    :math:`u=\\omega^2`.
    """
    loop = Loop(np.array([1.0]), np.array([1.0, 3.0, 2.0, 0.0]))
    result = loop_margins(loop, "textbook")

    assert result.stable
    assert min(result.phase_crossover_rad_s) == pytest.approx(np.sqrt(2.0), rel=1e-9)
    assert result.gain_margin_up_db == pytest.approx(20.0 * np.log10(6.0), abs=1e-6)
    assert not np.isfinite(result.gain_margin_down_db)

    wc = np.sqrt(max(np.real(np.roots([1.0, 5.0, 4.0, -1.0]))))
    assert result.gain_crossover_rad_s == pytest.approx(wc, rel=1e-9)
    expected_pm = 90.0 - np.degrees(np.arctan(wc)) - np.degrees(np.arctan(wc / 2.0))
    assert result.phase_margin_deg == pytest.approx(expected_pm, abs=1e-6)


def test_extractor_reproduces_an_analytic_double_integrator_pd_loop():
    """A PD-controlled double integrator: crossover and phase margin in closed form.

    :math:`L(s) = (K_d s + K_p)/(Js^2)` crosses unity gain at the positive root
    of :math:`J^2\\omega^4 = K_d^2\\omega^2 + K_p^2` and its phase margin is
    :math:`\\arctan(K_d\\omega_c/K_p)`. Same structure as the flight loop with
    the integrator frozen, so it validates the path the saturated case takes.
    """
    inertia, kp, kd = 0.12, 4.4e-3, 3.1e-2
    loop = Loop(np.array([kd, kp]), np.array([inertia, 0.0, 0.0]))
    result = loop_margins(loop, "pd")

    wc = np.sqrt(
        (kd**2 + np.sqrt(kd**4 + 4.0 * inertia**2 * kp**2)) / (2.0 * inertia**2)
    )
    assert result.gain_crossover_rad_s == pytest.approx(wc, rel=1e-9)
    assert result.phase_margin_deg == pytest.approx(
        np.degrees(np.arctan2(kd * wc, kp)), abs=1e-6
    )
    # A type-2 loop with a stabilising zero never reaches -180: no crossing in
    # either direction, which the extractor must report as unbounded rather than
    # as a missing measurement.
    assert not np.isfinite(result.gain_margin_up_db)
    assert not np.isfinite(result.gain_margin_down_db)
    assert result.phase_crossover_rad_s == ()


def test_disk_margin_matches_its_closed_form_on_a_constant_gain():
    """:math:`L=k` gives :math:`\\alpha = 2(1+k)/|1-k|` exactly [seiler2020]."""
    for k in (0.25, 3.0, 9.0):
        alpha, _, phase_deg = disk_margin(np.full(16, complex(k)))
        assert alpha == pytest.approx(2.0 * (1.0 + k) / abs(1.0 - k), rel=1e-12)
        assert phase_deg == pytest.approx(
            np.degrees(2.0 * np.arctan(alpha / 2.0)), rel=1e-12
        )


def test_an_off_window_gain_set_fails_loudly_instead_of_inflating(vehicle):
    """A crossover outside the sweep must raise, not report an unbounded margin.

    The grid bounds are constants rather than derived from the gains, so a
    configuration whose crossover falls outside them would be scored on a window
    that never saw it — and an unseen crossing reads as ``inf``, which is
    indistinguishable from "no crossing exists" and is therefore a *better*
    margin than the design has. The guard is only worth having if it fires, so
    both directions are driven here.
    """
    # Crossover scales as Kd/J, so the multipliers are written against the
    # vehicle's own committed Kd rather than as absolute gains — the reference
    # tuning moved by three orders with the Push 60 re-baseline and a transcribed
    # gain would have quietly stopped landing where this test needs it. A 1e6x
    # derivative gain puts the crossover far above the 1e4 rad/s top of the
    # continuous sweep.
    fast = replace(
        vehicle,
        pid=replace(vehicle.pid, kd_nm_per_radps=1.0e6 * vehicle.pid.kd_nm_per_radps),
    )
    with pytest.raises(GridWindowError, match="no unity-gain crossing"):
        axis_margins(fast, 0, sampled=False)

    # And a crossing that lands *inside* but within a decade of an edge is the
    # same hazard one step earlier: the next configuration falls off.
    marginal = replace(
        vehicle,
        pid=replace(vehicle.pid, kd_nm_per_radps=1.0e3 * vehicle.pid.kd_nm_per_radps),
    )
    with pytest.raises(GridWindowError, match="within 1 decade of the sweep edge"):
        axis_margins(marginal, 0, sampled=False)

    # The committed configuration is comfortably inside, or the guard would be
    # failing the vehicle it is meant to protect.
    assert axis_margins(vehicle, 0, sampled=False).stable


def test_continuous_loop_matches_the_hand_derived_transfer_function(vehicle):
    """`open_loop` is (Kd s^2 + Kp s + Ki)/(J s^3), coefficient for coefficient."""
    for i in range(3):
        loop = open_loop(vehicle, i)
        inertia = vehicle.principal_moments_kgm2[i]
        assert loop.num == pytest.approx(
            [
                vehicle.pid.kd_nm_per_radps,
                vehicle.pid.kp_nm_per_rad,
                vehicle.pid.ki_nm_per_rad_s,
            ]
        )
        assert loop.den == pytest.approx([inertia, 0.0, 0.0, 0.0])
        assert not loop.discrete


def test_the_sampled_loop_has_no_spurious_pole_at_unity(vehicle):
    """The state-space route must not leave the ``(z-1)`` factor the paths share.

    Summing the three feedback paths as transfer functions leaves a pole-zero
    pair at ``z = 1`` that reads as marginal instability. Three open-loop poles,
    all at ``z = 1``, is the correct realisation.
    """
    for i in range(3):
        loop = discrete_open_loop(vehicle, i)
        assert loop.discrete
        assert loop.dt == pytest.approx(vehicle.control_period_s)
        # A triple root is only recoverable to about eps^(1/3), so 1e-4 is the
        # tight tolerance here, not a loose one. A spurious fourth pole — the
        # defect this guards — would be a whole extra root, not a rounding.
        assert loop.den.size == 4
        assert np.roots(loop.den) == pytest.approx(np.ones(3), abs=1e-4)
        assert loop.is_stable()


# ── The vehicle ──────────────────────────────────────────────────────────────


@pytest.mark.verifies("REQ-ACTL-006")
def test_committed_gains_meet_the_margin_requirement(vehicle, record_property):
    """Every axis clears 6 dB / 30° on the sampled-data loop at the committed gains."""
    report = margin_report(vehicle, sampled=True)
    assert report.sampled
    assert report.failures() == []

    worst_phase = min(a.phase_margin_deg for a in report.axes)
    record_property("margin_pct", f"{100.0 * worst_phase / MIN_PHASE_MARGIN_DEG:.0f}")

    for axis in report.axes:
        assert axis.stable
        assert axis.sample_period_s == pytest.approx(vehicle.control_period_s)
        assert axis.gain_margin_db >= MIN_GAIN_MARGIN_DB
        assert axis.phase_margin_deg >= MIN_PHASE_MARGIN_DEG
        assert axis.sensitivity_peak <= MAX_SENSITIVITY_PEAK
        # The design also clears the preferred target, which is worth asserting
        # rather than only reporting: it is the difference between a design that
        # scrapes the floor and one with room for a gain re-tune.
        assert axis.phase_margin_deg >= PREFERRED_PHASE_MARGIN_DEG

    assert report.passes


@pytest.mark.verifies("REQ-ACTL-006")
def test_margins_survive_a_one_cycle_computation_delay(vehicle):
    """A pessimistic extra 100 ms of transport delay does not break the requirement."""
    report = margin_report(vehicle, computation_delay_cycles=1)
    assert report.failures() == []
    for axis in report.axes:
        assert axis.phase_margin_deg >= MIN_PHASE_MARGIN_DEG
        # The delay must actually cost phase, or the case tests nothing.
        nominal = axis_margins(vehicle, "xyz".index(axis.axis))
        assert axis.phase_margin_deg < nominal.phase_margin_deg


def test_sampling_costs_phase_relative_to_the_continuous_idealisation(vehicle):
    """The ZOH lag is real but small here, because the crossover is well under Nyquist."""
    for i in range(3):
        sampled = axis_margins(vehicle, i, sampled=True)
        continuous = axis_margins(vehicle, i, sampled=False)
        assert sampled.phase_margin_deg < continuous.phase_margin_deg
        expected_loss_deg = np.degrees(
            continuous.gain_crossover_rad_s * vehicle.control_period_s / 2.0
        )
        loss = continuous.phase_margin_deg - sampled.phase_margin_deg
        assert loss == pytest.approx(expected_loss_deg, rel=0.25)
        # "Small" is stated against the margin it eats into rather than as an
        # absolute number of degrees: the absolute loss is w_c * T / 2 and moves
        # with the bandwidth (0.3 deg at the 12 kg tuning's 0.27 rad/s crossover,
        # 3.4 deg at this one's 1.26 rad/s), while the claim being made — that
        # sampling is not what decides the design — is a ratio.
        assert loss < 0.1 * sampled.phase_margin_deg


def test_the_loop_is_conditionally_stable_and_says_so(vehicle):
    """The gain margin of this design is a *downward* one; report both directions."""
    for i in range(3):
        result = axis_margins(vehicle, i)
        assert result.conditionally_stable
        assert np.isfinite(result.gain_margin_down_db)
        # Both -180 crossings must be found, and classified in opposite
        # directions — this is the multiple-crossing path, on the real loop
        # rather than on a synthetic fixture. The low-frequency one is at
        # sqrt(Ki/Kd) where |L| > 1 (the downward margin); the sampled loop
        # touches the negative real axis again at Nyquist, where |L| < 1 (the
        # upward one). Taking only the first would report one and miss the other.
        assert len(result.phase_crossover_rad_s) == 2
        expected = np.sqrt(vehicle.pid.ki_nm_per_rad_s / vehicle.pid.kd_nm_per_radps)
        assert min(result.phase_crossover_rad_s) == pytest.approx(expected, rel=0.02)
        assert max(result.phase_crossover_rad_s) == pytest.approx(
            np.pi / vehicle.control_period_s, rel=1e-9
        )
        assert np.isfinite(result.gain_margin_up_db)
        assert result.gain_margin_db == min(
            result.gain_margin_up_db, result.gain_margin_down_db
        )
        # And the loop really is destabilised by that much gain reduction: build
        # the reduced-gain loop and check its closed-loop poles, rather than
        # trusting the margin arithmetic that produced the number.
        loop = discrete_open_loop(vehicle, i)
        reduced = 10.0 ** (-result.gain_margin_down_db / 20.0)
        assert loop.scaled(reduced * 1.02).is_stable()
        assert not loop.scaled(reduced * 0.98).is_stable()


def test_freezing_the_integrator_removes_the_conditional_stability(vehicle):
    """The saturated-loop case (Ki frozen) is a plain type-2 loop with no down-margin."""
    for i in range(3):
        result = axis_margins(vehicle, i, integrator=False)
        assert result.stable
        assert not np.isfinite(result.gain_margin_down_db)
        assert result.phase_margin_deg >= MIN_PHASE_MARGIN_DEG


def test_the_gate_can_fail(vehicle):
    """A deliberately de-tuned derivative gain must be reported, not absorbed.

    Without this the pass in the headline test says nothing about whether the
    threshold is applied at all.
    """
    weak = replace(
        vehicle,
        pid=PidGains(
            kp_nm_per_rad=vehicle.pid.kp_nm_per_rad,
            ki_nm_per_rad_s=vehicle.pid.ki_nm_per_rad_s,
            kd_nm_per_radps=vehicle.pid.kd_nm_per_radps / 12.0,
            max_integral_rad_s=vehicle.pid.max_integral_rad_s,
            max_torque_nm=vehicle.pid.max_torque_nm,
        ),
    )
    report = margin_report(weak)
    assert not report.passes
    assert any("phase margin" in f or "unstable" in f for f in report.failures())

    structured = control_analysis_report(weak, "synthetic-detuned")
    assert not structured.passes
    assert any(c.requirement == "REQ-ACTL-006" for c in structured.failures())
