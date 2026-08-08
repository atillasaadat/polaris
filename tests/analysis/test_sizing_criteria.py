"""Wheel, magnetorquer and disturbance sizing verdicts (design doc §5.3, §7, §12).

Every criterion is exercised on **both sides**: the reference vehicle as
committed, and a deliberately perturbed copy that flips the verdict. A criterion
only ever seen passing is a criterion nobody has shown can fail.

Assertions are on the structured :class:`analysis.common.report.AnalysisReport`,
never on rendered text or figures, per ``analysis/CLAUDE.md``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.disturbances import (
    disturbance_budget,
    exponential_density,
    field_statistics,
    gravity_gradient_torque,
    srp_torque,
)
from analysis.sizing.report import sizing_analysis, sizing_report


def _criterion(report, fragment):
    """The one criterion whose name contains @p fragment."""
    matches = [c for c in report.criteria if fragment in c.name]
    assert len(matches) == 1, f"{fragment!r} matched {[c.name for c in matches]}"
    return matches[0]


# --------------------------------------------------------------------------
# The disturbance budget
# --------------------------------------------------------------------------


def test_the_density_comes_from_the_plants_own_band_table():
    """Parsed from ``sim/world/atmosphere.cpp``, not transcribed.

    At a band's base altitude the exponential is exactly the base density, so
    this pins the parse against a value the C++ file itself carries. A
    transcription would be a copy, and every copy is a place for the analysis
    and the plant to disagree about the vehicle's own environment.
    """
    assert exponential_density(500_000.0) == pytest.approx(6.967e-13)
    assert exponential_density(400_000.0) == pytest.approx(3.725e-12)
    # Between bands it decays with the band's scale height, monotonically.
    assert exponential_density(450_000.0) > exponential_density(550_000.0)
    with pytest.raises(ValueError, match="below the band table"):
        exponential_density(-1.0)


def test_gravity_gradient_carries_the_three_halves_and_not_a_three(vehicle):
    """:math:`\\tfrac32 n^2\\Delta I`, the maximum of :math:`\\sin 2\\theta` being 1.

    Dropping the half is the standard doubling error in this formula, so the
    closed form is pinned rather than trusted. Both the inertia spread and the
    mean motion are read from the vehicle: the magnitude moved by ~30x in the
    50 kg re-baseline while the *identity* being checked did not, and a pinned
    magnitude would only have recorded which vehicle was current.
    """
    n = vehicle.orbit.mean_motion_rad_s
    moments = vehicle.principal_moments_kgm2
    spread = float(np.max(moments) - np.min(moments))
    expected = 1.5 * n**2 * spread
    assert gravity_gradient_torque(vehicle) == pytest.approx(expected, rel=1e-6)
    # The half is what this test exists for: the 3n²ΔI form would double it.
    assert gravity_gradient_torque(vehicle) != pytest.approx(
        3.0 * n**2 * spread, rel=1e-6
    )


def test_srp_uses_the_configs_own_reflectivity_and_lever_arm(vehicle):
    """:math:`(\\Phi/c) A C_r |d|` with every factor read from the config."""
    expected = (
        (1361.0 / 2.99792458e8)
        * vehicle.srp_area_m2
        * vehicle.srp_cr
        * float(np.linalg.norm(vehicle.cp_offset_srp_m))
    )
    assert srp_torque(vehicle) == pytest.approx(expected)


def test_the_budget_totals_match_the_configs_own_stated_environment(vehicle):
    """The budget is internally consistent and sits in the physical band.

    The absolute total used to be pinned at the 12 kg bus's ~2e-7 N·m. That
    number is a property of the vehicle's *areas and lever arms*, not of this
    package, and it moved by an order of magnitude in the 50 kg re-baseline. The
    durable claims are that the split accounts for the whole total, that every
    term is positive and finite, and that the total stays inside the band a LEO
    smallsat can physically occupy — a decade either side of 1e-6 N·m. A number
    outside that indicts the config or the formulas; a number inside it is not
    evidence of much, which is why the term-by-term closed forms are pinned
    individually above.
    """
    budget = disturbance_budget(vehicle)
    assert budget.secular_nm + budget.cyclic_nm == pytest.approx(budget.total_nm)
    assert 1.0e-8 < budget.total_nm < 1.0e-4
    assert budget.secular_nm > 0.0 and budget.cyclic_nm > 0.0
    # The field the terms were evaluated in matches the orbit the config states:
    # the IGRF magnitude over this 500 km SSO runs ~22-52 uT.
    assert 2.0e-5 < budget.field.min_t < 2.6e-5
    assert 4.4e-5 < budget.field.max_t < 5.2e-5


def test_the_secular_cyclic_split_follows_the_stated_assumption(vehicle):
    """Changing the assumption moves the torque between the halves, nothing else."""
    lvlh = disturbance_budget(vehicle)
    inertial = disturbance_budget(
        vehicle,
        SizingAssumptions(secular_fraction_aero=1.0, secular_fraction_srp=0.0),
    )
    assert inertial.total_nm == pytest.approx(lvlh.total_nm)
    assert inertial.secular_nm > lvlh.secular_nm
    assert inertial.cyclic_nm < lvlh.cyclic_nm


def test_the_field_statistics_bracket_the_mean(vehicle):
    """min <= mean <= max over a sampled orbit, from the committed IAGA table."""
    stats = field_statistics(vehicle, samples=181)
    assert stats.min_t <= stats.mean_t <= stats.max_t
    assert stats.samples == 181


# --------------------------------------------------------------------------
# Wheel sizing
# --------------------------------------------------------------------------


def test_d1_is_reported_but_not_judged_when_the_rods_can_detumble(
    vehicle, reference_config
):
    """Whether the wheels alone must catch the raw tip-off is a CONOPS question.

    A vehicle that detumbles magnetically flies rods-then-wheels, so the binding
    wheel requirement is the D1b handover; judging D1 as well would size the
    wheels for a mode the vehicle never flies. D1 stays in the table either way,
    because "what if the rods are lost" is a real question and the number
    answers it — it simply carries no verdict while M2 passes.
    """
    report = sizing_report(vehicle, reference_config)
    assert _criterion(report, "M2 detumble authority").passes
    judged = [c.name for c in report.criteria]
    assert not any("usable momentum vs D1 tip-off" in n for n in judged)
    # D1b, the handover that does bind, is judged.
    assert _criterion(report, "usable momentum vs D1b").passes


def test_d1_binds_when_the_rods_cannot_detumble(vehicle, reference_config):
    """Strip the magnetic authority and the wheels inherit the whole tip-off.

    The other half of the coupling above, and the reason it is a coupling rather
    than a blanket exemption: with no usable rods there is no detumble phase, so
    D1 becomes a hard requirement and is judged again.
    """
    feeble = dataclasses.replace(vehicle, mtq_max_dipole_am2=1.0e-6)
    report = sizing_report(feeble, reference_config)
    assert not _criterion(report, "M2 detumble authority").passes
    d1 = _criterion(report, "usable momentum vs D1 tip-off")
    assert d1.threshold == pytest.approx(
        1.3 * float(np.max(vehicle.principal_moments_kgm2)) * np.deg2rad(5.0), rel=1e-6
    )


def test_the_oversizing_check_fires_on_a_wheel_far_above_its_drivers(
    vehicle, reference_config
):
    """A capability far above every driver is a finding, not a large margin.

    Written against a **constructed** vehicle rather than the committed one.
    This test used to assert that the reference wheels were 78x oversized, which
    was true of the 12 kg bus and is exactly the defect the tool was built to
    expose; the 50 kg re-baseline fixed it, and a test pinning the broken state
    fails on the repair. What is durable is that the check fires when the
    condition holds, so the condition is manufactured here.
    """
    report = sizing_report(vehicle, reference_config)
    oversizing = _criterion(report, "oversizing factor")
    assert oversizing.sense == "max"
    assert oversizing.passes  # the committed design is right-sized

    absurd = dataclasses.replace(
        vehicle, wheel_max_momentum_nms=100.0 * vehicle.wheel_max_momentum_nms
    )
    fired = _criterion(sizing_report(absurd, reference_config), "oversizing factor")
    assert not fired.passes
    assert fired.measured > oversizing.measured

    right_sized = dataclasses.replace(vehicle, wheel_max_momentum_nms=0.01)
    assert _criterion(
        sizing_report(right_sized, reference_config), "oversizing factor"
    ).passes


def test_wheel_torque_is_judged_on_the_guaranteed_radius_not_the_body_axis(
    vehicle, reference_config
):
    """The guaranteed radius, not the body-axis figure a per-axis check would claim.

    For a four-wheel pyramid the number that survives an *arbitrary* demand
    direction is ``4·τ/√6``; the body-axis reach is ``4·τ/√3``, larger by 41 %
    in the unconservative direction. Both are written against the vehicle's own
    per-wheel torque so the identity is what is pinned, not the magnitude — the
    wheel changed from RW-X to RW-S in Push 60 and the ratio did not.
    """
    report = sizing_report(vehicle, reference_config)
    torque = _criterion(report, "wheel torque")
    tau = vehicle.wheel_max_torque_nm
    assert torque.passes
    assert torque.measured == pytest.approx(4.0 * tau / np.sqrt(6.0), rel=1e-3)
    assert torque.measured < 4.0 * tau / np.sqrt(3.0)

    # A wheel too weak by construction: a quarter of what the design needs,
    # scaled off the vehicle rather than a literal that stops biting on a
    # smaller wheel.
    weak = dataclasses.replace(vehicle, wheel_max_torque_nm=0.25 * tau)
    assert not _criterion(sizing_report(weak, reference_config), "wheel torque").passes


def test_the_commanded_torque_limit_may_not_exceed_the_installed_wheel(
    vehicle, reference_config
):
    """WheelMaxTorqueNm against the catalog's max_torque_nm — two numbers, one bolt.

    The flight parameter and the hardware capability are set independently, so a
    wheel swap can leave the FSW authorised to command torque the unit cannot
    produce. That happened on this vehicle (RW-X 0.025 N·m → RW-S 0.002 N·m) and
    nothing caught it; this is what catches it now.
    """
    report = sizing_report(vehicle, reference_config)
    commanded = _criterion(report, "commanded torque limit")
    assert commanded.sense == "max"
    assert commanded.passes
    assert commanded.measured == vehicle.wheel_max_torque_nm
    assert commanded.threshold == vehicle.wheel_catalog_torque_nm

    # Raise the flight limit past the catalog: derived from the vehicle, so this
    # keeps biting whatever wheel is installed next.
    optimistic = dataclasses.replace(
        vehicle, wheel_max_torque_nm=2.0 * vehicle.wheel_catalog_torque_nm
    )
    assert not _criterion(
        sizing_report(optimistic, reference_config), "commanded torque limit"
    ).passes


def test_a_deep_derate_warns_rather_than_fails(vehicle, reference_config):
    """A flight limit below the catalog value is legitimate; a big gap is a finding."""
    assert not [
        w for w in sizing_report(vehicle, reference_config).warnings if "derate" in w
    ]

    derated = dataclasses.replace(
        vehicle, wheel_max_torque_nm=0.1 * vehicle.wheel_catalog_torque_nm
    )
    report = sizing_report(derated, reference_config)
    assert _criterion(report, "commanded torque limit").passes
    assert [w for w in report.warnings if "derate" in w]


def test_the_slew_driver_is_reported_not_judged_without_a_commanded_rate(
    vehicle, reference_config
):
    """No slew rate in the config means no verdict on slew agility.

    A PASS that cannot fail is not a verdict. Supplying a rate makes the driver
    judged, which is the behaviour that lets a user with a slew requirement gate
    on it.
    """
    report = sizing_report(vehicle, reference_config)
    assert not [c for c in report.criteria if "D4" in c.name]

    analysis = sizing_analysis(vehicle)
    j_max = float(np.max(vehicle.principal_moments_kgm2))
    supported = analysis.wheels.supported_slew_radps
    assert supported == pytest.approx(analysis.wheels.usable_momentum_nms / j_max)

    # Demand twice what the envelope supports: a failure by construction on any
    # vehicle, where the old literal (1 deg/s) only failed on the 12 kg one.
    demanding = SizingAssumptions(slew_rate_radps=2.0 * supported)
    judged = sizing_report(vehicle, reference_config, demanding)
    assert not _criterion(judged, "D4 slew agility").passes

    # And half of it must pass, or the criterion is not measuring the envelope.
    modest = SizingAssumptions(slew_rate_radps=0.5 * supported / 1.3)
    assert _criterion(
        sizing_report(vehicle, reference_config, modest), "D4 slew agility"
    ).passes


# --------------------------------------------------------------------------
# Magnetorquer sizing
# --------------------------------------------------------------------------


def test_desaturation_authority_beats_the_secular_disturbance(
    vehicle, reference_config
):
    """M1 passes by four orders of magnitude, and fails when the physics says it should.

    The failing side is reached by making the secular disturbance large — a
    vehicle with a metre-class SRP lever arm — rather than by shrinking the
    threshold, so the criterion is shown responding to the quantity it is written
    on.
    """
    report = sizing_report(vehicle, reference_config)
    m1 = _criterion(report, "M1 desaturation authority")
    assert m1.passes

    unbalanced = dataclasses.replace(
        vehicle, cp_offset_srp_m=np.array([50.0, 0.0, 0.0]), srp_area_m2=60.0
    )
    assert not _criterion(
        sizing_report(unbalanced, reference_config), "M1 desaturation authority"
    ).passes


def test_the_rod_authority_uses_the_minimum_field_and_the_duty_factor(vehicle):
    """``eta * m_in * |B|_min * duty`` — every factor present and honest.

    Sizing at the orbit mean would overstate the authority by ~50 % on this
    orbit, and dropping the 2/3 cross-product efficiency by another 50 %.
    """
    analysis = sizing_analysis(vehicle)
    mtq = analysis.mtq
    assert mtq.dipole.inscribed == pytest.approx(15.0)  # orthogonal triad: the rating
    expected = (2.0 / 3.0) * 15.0 * analysis.budget.field.min_t * 0.5
    assert mtq.average_torque_nm == pytest.approx(expected)
    assert mtq.average_torque_nm < mtq.peak_torque_nm


def test_the_bdot_noise_floor_is_above_the_committed_exit_threshold(
    vehicle, reference_config
):
    """M3: the exit threshold must clear ``σ√2/(Δt·|B|)``, B-dot's own noise floor.

    This is the defect the tool was built to catch — on the 12 kg bus the
    committed threshold sat *below* the floor, so "detumble complete" was a
    coin flip on magnetometer noise. The re-baseline cleared it, so the test
    now asserts the **rule** and manufactures the violation rather than
    depending on the shipped vehicle still being wrong.
    """
    report = sizing_report(vehicle, reference_config)
    m3 = _criterion(report, "M3 DetumbleExitRadps")
    assert m3.passes
    assert m3.measured >= m3.threshold

    # Halve the threshold to put it back under the floor: the criterion must fire.
    under = dataclasses.replace(
        vehicle, detumble_exit_radps=0.5 * m3.threshold * np.pi / 180.0
    )
    assert not _criterion(
        sizing_report(under, reference_config), "M3 DetumbleExitRadps"
    ).passes

    quiet = dataclasses.replace(vehicle, mag_noise_t=1.0e-9)
    assert _criterion(
        sizing_report(quiet, reference_config), "M3 DetumbleExitRadps"
    ).passes


def test_the_noise_floor_scales_as_the_physics_says(vehicle):
    """Linear in sigma, inverse in the differencing interval and in the field."""
    base = sizing_analysis(vehicle).mtq.noise_floor
    noisier = sizing_analysis(
        dataclasses.replace(vehicle, mag_noise_t=2.0 * vehicle.mag_noise_t)
    ).mtq.noise_floor
    slower = sizing_analysis(
        dataclasses.replace(vehicle, control_period_s=0.2)
    ).mtq.noise_floor
    assert noisier.rate_worst_radps == pytest.approx(2.0 * base.rate_worst_radps)
    assert slower.rate_worst_radps == pytest.approx(0.5 * base.rate_worst_radps)
    assert base.rate_mean_radps < base.rate_worst_radps


# --------------------------------------------------------------------------
# The report as a whole
# --------------------------------------------------------------------------


def test_no_criterion_claims_a_requirement_it_does_not_verify(
    vehicle, reference_config
):
    """Sizing has no requirement in the baseline, and the report says so.

    Only the momentum-envelope criterion carries an ID, because REQ-ACTL-009 is
    genuinely written on that bound. Everything else is a design check with no
    requirement behind it, and borrowing an ID would manufacture verification
    evidence.
    """
    report = sizing_report(vehicle, reference_config)
    ids = {c.requirement for c in report.criteria if c.requirement}
    assert ids == {"REQ-ACTL-009"}
    assert any("No requirement in the baseline" in w for w in report.warnings)


def test_the_assumptions_block_carries_every_assumed_input(vehicle, reference_config):
    """A margin without its assumptions is not a result — so they are all rendered."""
    report = sizing_report(vehicle, reference_config)
    text = " ".join(report.assumptions)
    for expected in ("Tip-off rate", "Desaturation interval", "LVLH", "30%"):
        assert expected in text


def test_the_incompatible_bounds_conflict_is_raised_when_it_exists(
    vehicle, reference_config
):
    """When no exit threshold satisfies both bounds, that is said out loud.

    The finding the tool exists to produce: B-dot cannot certify a rate below
    its own noise floor, and the wheels must be able to take the handover at
    whatever rate it does certify. When the certified envelope is too small for
    the floor's momentum, *no* ``DetumbleExitRadps`` satisfies both — a design
    fault, not a tuning one, and it must surface as a warning plus a failing
    criterion rather than a silent recommendation.

    The committed vehicle satisfies both (the 50 kg re-baseline is what closed
    it), so the conflict is manufactured by shrinking the certified envelope.
    Asserting it on the shipped config would mean asserting the design is
    broken — which was true when this tool was written and is the state it was
    built to remove.
    """
    healthy = sizing_report(vehicle, reference_config)
    assert _criterion(healthy, "handover at the B-dot noise floor").passes
    assert not any("incompatible" in w for w in healthy.warnings)

    cramped = dataclasses.replace(vehicle, momentum_envelope_nms=1.0e-3)
    report = sizing_report(cramped, reference_config)
    assert not _criterion(report, "handover at the B-dot noise floor").passes
    assert any("incompatible" in w for w in report.warnings)
