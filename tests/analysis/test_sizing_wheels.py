"""The wheel sizing drivers and what they are judged against (design doc §7, §12).

:mod:`tests.analysis.test_sizing_criteria` covers the verdicts. This covers the
arithmetic underneath them: each of D1, D1b, D2, D3 and D4 against its closed
form on a vehicle whose inputs are chosen here, the two capability figures
(hardware :math:`r_{\\mathrm{in}}` and the usable envelope) and the gap between
them, and the two "largest driver" notions that are deliberately different —
``largest_driver`` weighs only what is judged, ``largest_demand`` weighs every
real demand, and the oversizing verdict must use the second.

Nothing here asserts a momentum in N·m·s that belongs to the committed bus. Where
the reference vehicle is used, the expectation is built from ``vehicle.*`` fields
and the assumptions object, so a re-baseline moves both sides together.

References
----------
Wertz, Everett & Puschell §19.2 [wertz2011] — the sizing drivers; Markley &
Crassidis §7.3 [markley2014] — wheel arrays; design doc §8.5 for the envelope.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.disturbances import disturbance_budget
from analysis.sizing.envelope import DegenerateArrayError
from analysis.sizing.report import sizing_analysis
from analysis.sizing.wheels import (
    CYCLIC_RMS_FACTOR,
    MomentumDriver,
    criteria,
    wheel_sizing,
)


@pytest.fixture()
def budget(vehicle):
    """The reference vehicle's disturbance budget, shared by the driver tests."""
    return disturbance_budget(vehicle)


def _driver(sizing, prefix):
    """The one driver whose name starts with @p prefix, e.g. ``"D2"``."""
    return next(d for d in sizing.drivers if d.name.startswith(prefix))


def _criterion(criteria_list, fragment):
    matches = [c for c in criteria_list if fragment in c.name]
    assert len(matches) == 1, f"{fragment!r} matched {[c.name for c in matches]}"
    return matches[0]


# --------------------------------------------------------------------------
# The five drivers, each against its closed form
# --------------------------------------------------------------------------


def test_d1_and_d1b_are_the_worst_axis_momentum_at_their_two_rates(vehicle, budget):
    """:math:`h = J_{\\max}\\omega`, at the tip-off rate and at the B-dot handover.

    The **largest** principal moment is the right one: the driver is the momentum
    the array must hold about the worst axis, and using the mean would understate
    it by whatever the inertia spread happens to be. The two rates are the
    assumption's tip-off and the config's own ``DetumbleExitRadps``, so the pair
    also pins which rate belongs to which driver — swapping them is a silent
    factor of tens.
    """
    assumptions = SizingAssumptions(tipoff_rate_radps=np.deg2rad(4.0))
    sizing = wheel_sizing(vehicle, budget, assumptions)
    j_max = float(np.max(vehicle.principal_moments_kgm2))
    assert _driver(sizing, "D1 ").required_nms == pytest.approx(j_max * np.deg2rad(4.0))
    assert _driver(sizing, "D1b").required_nms == pytest.approx(
        j_max * vehicle.detumble_exit_radps
    )


def test_d2_is_the_quarter_period_integral_with_its_peak_to_rms_factor(vehicle, budget):
    """:math:`h_{D2} = 0.707\\,\\tau_{cyc}T/4`, and it is 11 % above the exact integral.

    A sinusoidal torque :math:`\\tau\\sin nt` integrates to exactly
    :math:`\\tau/n = 0.159\\,\\tau T` over a quarter period; the SMAD sizing form
    is :math:`0.177\\,\\tau T`. The module says it takes the conservative one, so
    the ratio to the exact integral is pinned at :math:`2\\pi\\times0.707/4`
    rather than left as a comment nobody checks.
    """
    sizing = wheel_sizing(vehicle, budget)
    period = vehicle.orbit.period_s
    d2 = _driver(sizing, "D2")
    assert d2.required_nms == pytest.approx(
        CYCLIC_RMS_FACTOR * budget.cyclic_nm * period / 4.0
    )
    exact_quarter_period = budget.cyclic_nm / vehicle.orbit.mean_motion_rad_s
    assert d2.required_nms / exact_quarter_period == pytest.approx(
        2.0 * np.pi * CYCLIC_RMS_FACTOR / 4.0, rel=1e-9
    )
    assert d2.required_nms > exact_quarter_period


def test_d3_is_the_secular_torque_times_the_desaturation_interval(vehicle, budget):
    """:math:`h_{D3} = \\tau_{sec}T_{desat}`, linear in the operational choice.

    The desaturation interval is the single biggest lever on required wheel
    momentum and it is an operations decision, not a physical one — so the
    linearity is asserted directly: halving the interval must halve the driver.
    The default is one orbit, which is checked too, because a default that
    quietly became something else would rescale every wheel margin.
    """
    default = wheel_sizing(vehicle, budget)
    assert _driver(default, "D3").required_nms == pytest.approx(
        budget.secular_nm * vehicle.orbit.period_s
    )

    brisk = wheel_sizing(
        vehicle,
        budget,
        SizingAssumptions(desat_interval_s=0.5 * vehicle.orbit.period_s),
    )
    assert _driver(brisk, "D3").required_nms == pytest.approx(
        0.5 * _driver(default, "D3").required_nms
    )


def test_d4_is_judged_only_when_a_slew_rate_is_declared(vehicle, budget):
    """:math:`h_{D4} = J_{\\max}\\omega_{slew}` — zero and unjudged without one.

    A PASS that cannot fail is not a verdict, so a config with no slew
    requirement gets a reported diagnostic instead. The declared case must still
    compute the right momentum, or the option is decoration.
    """
    silent = _driver(wheel_sizing(vehicle, budget), "D4")
    assert not silent.judged
    assert silent.required_nms == 0.0
    assert "no commanded slew rate" in silent.inputs

    rate = np.deg2rad(0.5)
    declared = _driver(
        wheel_sizing(vehicle, budget, SizingAssumptions(slew_rate_radps=rate)), "D4"
    )
    assert declared.judged
    assert declared.required_nms == pytest.approx(
        float(np.max(vehicle.principal_moments_kgm2)) * rate
    )


def test_d1_stops_being_judged_once_the_rods_can_remove_the_tip_off(vehicle, budget):
    """The CONOPS switch: rods-then-wheels judges D1b, no rods judges D1.

    Both calls produce the same *number* for D1 — the vehicle's tip-off momentum
    does not depend on how it is removed — and differ only in whether a criterion
    is written on it. Pinning the number's invariance is the point: the switch
    must change the verdict logic and nothing else, so "what if the rods are
    lost" stays answerable from the table either way.
    """
    with_rods = _driver(wheel_sizing(vehicle, budget, magnetic_detumble=True), "D1 ")
    without = _driver(wheel_sizing(vehicle, budget, magnetic_detumble=False), "D1 ")
    assert with_rods.required_nms == pytest.approx(without.required_nms)
    assert without.judged and not with_rods.judged
    assert "rods remove the tip-off" in with_rods.inputs

    judged_names = [c.name for c in criteria(wheel_sizing(vehicle, budget, None, True))]
    assert not any("usable momentum vs D1 tip-off" in n for n in judged_names)
    # The hardware criterion is written on D1 regardless, so the question "can
    # the wheels themselves catch a raw tip-off" always has an answer on record.
    assert any("hardware momentum vs D1 tip-off" in n for n in judged_names)


# --------------------------------------------------------------------------
# The two capability figures
# --------------------------------------------------------------------------


def test_the_two_envelopes_are_built_on_the_two_different_capacities(vehicle, budget):
    """Momentum from the catalog, torque from the flight parameter — not one number.

    The momentum envelope is the wheel's physical storage (``max_momentum_nms``
    from the hardware catalog) while the torque envelope is what the allocator is
    *authorised* to command (``WheelMaxTorqueNm``, a flight parameter). Reading
    both from one source would hide exactly the mismatch the commanded-torque
    criterion exists to catch.
    """
    sizing = wheel_sizing(vehicle, budget)
    assert sizing.momentum.capacity == vehicle.wheel_max_momentum_nms
    assert sizing.torque.capacity == vehicle.wheel_max_torque_nm
    assert sizing.commanded_torque_nm == vehicle.wheel_max_torque_nm
    assert sizing.catalog_torque_nm == vehicle.wheel_catalog_torque_nm


def test_the_usable_envelope_is_the_smaller_of_hardware_and_flight_ceiling(
    vehicle, budget
):
    """``usable = min(r_in, MomentumEnvelopeNms)``, and the flag says which bound bit.

    Momentum the vehicle will raise an envelope event over is momentum it does
    not have, so every momentum criterion is judged on the smaller figure. Both
    sides are manufactured here — a certified ceiling far below the hardware and
    one far above it — because which one binds is a property of the design and
    has already flipped once on this vehicle.
    """
    hardware = wheel_sizing(vehicle, budget).momentum.inscribed

    cramped = dataclasses.replace(vehicle, momentum_envelope_nms=0.1 * hardware)
    limited = wheel_sizing(cramped, budget)
    assert limited.envelope_limited
    assert limited.usable_momentum_nms == pytest.approx(0.1 * hardware)

    generous = dataclasses.replace(vehicle, momentum_envelope_nms=10.0 * hardware)
    free = wheel_sizing(generous, budget)
    assert not free.envelope_limited
    assert free.usable_momentum_nms == pytest.approx(hardware)


def test_the_supported_slew_rate_is_the_usable_momentum_over_the_worst_axis(
    vehicle, budget
):
    """:math:`\\omega = h_{usable}/J_{\\max}` — the diagnostic, on the usable figure.

    Reported rather than judged, so nothing else guards it: computed from the
    hardware radius instead of the usable one it would overstate the agility of
    any envelope-limited vehicle, which is every vehicle whose ceiling comes from
    a SISO validity argument.
    """
    sizing = wheel_sizing(vehicle, budget)
    assert sizing.supported_slew_radps == pytest.approx(
        sizing.usable_momentum_nms / float(np.max(vehicle.principal_moments_kgm2))
    )


def test_the_required_torque_is_the_control_authority_plus_the_disturbance(
    vehicle, budget
):
    """``PidMaxTorqueNm + total disturbance``: the array must supply both at once.

    The disturbance does not pause while the controller slews, so the demands
    add. Using the secular half here — the tempting alternative — would size the
    array for the average environment rather than the worst one it must hold
    attitude in.
    """
    sizing = wheel_sizing(vehicle, budget)
    assert sizing.required_torque_nm == pytest.approx(
        vehicle.pid.max_torque_nm + budget.total_nm
    )
    assert sizing.required_torque_nm > vehicle.pid.max_torque_nm


# --------------------------------------------------------------------------
# largest_driver vs largest_demand — the distinction that bites
# --------------------------------------------------------------------------


def _with_drivers(sizing, *drivers):
    """@p sizing carrying a chosen driver set, so the two accessors can be separated."""
    return dataclasses.replace(sizing, drivers=tuple(drivers))


def test_oversizing_weighs_reported_drivers_and_not_only_judged_ones(vehicle, budget):
    """``largest_demand`` includes an unjudged driver; ``largest_driver`` does not.

    The distinction is the bug this pair exists to prevent. Oversizing asks about
    the *unit class* — is this the wrong wheel for this vehicle? — so it must
    weigh every real demand, including one that is reported rather than gated.
    Using only the judged set would call a wheel oversized the moment the
    criterion that justified its size stopped gating (D1, once the rods are shown
    to remove the tip-off), which is an artefact of the verdict logic rather than
    a fact about the hardware.
    """
    small_judged = MomentumDriver(
        name="judged", required_nms=1.0, formula="", inputs="", judged=True
    )
    large_reported = MomentumDriver(
        name="reported", required_nms=10.0, formula="", inputs="", judged=False
    )
    sizing = _with_drivers(wheel_sizing(vehicle, budget), small_judged, large_reported)
    assert sizing.largest_driver is small_judged
    assert sizing.largest_demand is large_reported
    assert sizing.oversizing == pytest.approx(sizing.momentum.inscribed / 10.0)

    verdict = _criterion(criteria(sizing), "oversizing factor")
    assert verdict.measured == pytest.approx(sizing.oversizing)
    assert "reported" in verdict.note


# --------------------------------------------------------------------------
# The criteria the drivers feed
# --------------------------------------------------------------------------


def test_every_judged_driver_gets_the_margin_factor_on_its_threshold(vehicle, budget):
    """Threshold = ``margin * required``, measured = the usable envelope.

    The 30 % convention is applied in exactly one place per criterion, so a
    driver's requirement and its threshold must differ by exactly the margin. The
    margin is read off the assumptions object rather than typed, which is what
    lets a caller size to a different convention without editing the package.
    """
    assumptions = SizingAssumptions(margin=1.5, slew_rate_radps=np.deg2rad(0.2))
    sizing = wheel_sizing(vehicle, budget, assumptions)
    rows = criteria(sizing, assumptions)
    for driver in sizing.drivers:
        if not driver.judged:
            continue
        row = _criterion(rows, f"usable momentum vs {driver.name}")
        assert row.threshold == pytest.approx(1.5 * driver.required_nms)
        assert row.measured == pytest.approx(sizing.usable_momentum_nms)
        assert row.sense == "min"
        assert row.units == "N.m.s"


def test_the_hardware_and_usable_momentum_criteria_differ_only_in_what_they_measure(
    vehicle, budget
):
    """Same D1 threshold, two capabilities — which is how a ceiling problem is named.

    A design whose only problem is its certified linear regime looks identical to
    one whose wheels are genuinely too small unless both are reported. The pair
    is manufactured with a cramped envelope so the two verdicts actually differ,
    which is the only state in which the distinction earns its place.
    """
    cramped = dataclasses.replace(
        vehicle,
        momentum_envelope_nms=1.0e-4 * wheel_sizing(vehicle, budget).momentum.inscribed,
    )
    rows = criteria(wheel_sizing(cramped, budget, magnetic_detumble=False))
    hardware = _criterion(rows, "hardware momentum vs D1")
    usable = _criterion(rows, "usable momentum vs D1 tip-off")
    assert hardware.threshold == pytest.approx(usable.threshold)
    assert hardware.measured > usable.measured
    assert hardware.passes and not usable.passes


def test_the_torque_criterion_is_the_guaranteed_radius_against_the_summed_demand(
    vehicle, budget
):
    """Measured is :math:`r_{\\mathrm{in}}` of the torque zonotope, never the body axis.

    Both numbers appear on the row — the guaranteed radius as the measurement and
    the per-body-axis reach in the note — because the difference between them is
    what an axis-only check would have silently claimed. On the reference
    four-wheel pyramid that is 41 % in the unconservative direction.
    """
    sizing = wheel_sizing(vehicle, budget)
    assumptions = SizingAssumptions()
    row = _criterion(criteria(sizing, assumptions), "wheel torque, guaranteed")
    assert row.measured == pytest.approx(sizing.torque.inscribed)
    assert row.threshold == pytest.approx(
        assumptions.margin * sizing.required_torque_nm
    )
    assert sizing.torque.inscribed < float(np.max(sizing.torque.per_body_axis))


def test_every_judged_driver_reaches_the_momentum_figure_with_its_label(vehicle):
    """Each driver arrow carries its short code at the tip and its name in the legend.

    Cheap, and it catches a class this package has already produced: the arrow's
    tip label is a ``str`` parameter that a local of a different type can shadow,
    which leaves the figure raising — or worse, labelling every driver the same —
    with nothing else in the suite looking at the trace text. Asserting the codes
    and the names are *present and distinct* is enough; nothing here asserts on
    pixels or on geometry, which is the report's job.
    """
    from analysis.sizing.interactive import momentum_envelope_figure

    analysis = sizing_analysis(vehicle)
    labels = {
        trace.text[-1]: trace.name
        for trace in momentum_envelope_figure(analysis).data
        if getattr(trace, "text", None)
    }
    judged = [d for d in analysis.wheels.drivers if d.judged]
    assert len(labels) == len(judged)
    for driver in judged:
        code = driver.name.split(" ")[0]
        assert code in labels, f"{driver.name} has no arrow on the momentum figure"
        assert code in labels[code]


def test_a_wheel_array_that_cannot_span_three_axes_is_refused(vehicle, budget):
    """Coplanar spin axes raise rather than returning a small margin.

    Three wheels in one plane give a vehicle with no authority about the normal;
    a radius reported for that layout would read as a tight margin on a report
    instead of as an uncontrollable axis. The message must name the geometry so
    the reader is not left guessing which of the axis parameters is wrong.
    """
    flat = dataclasses.replace(
        vehicle,
        wheel_spin_axes=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]).T,
    )
    with pytest.raises(DegenerateArrayError, match="coplanar"):
        wheel_sizing(flat, budget)

    single = dataclasses.replace(vehicle, wheel_spin_axes=np.array([[1.0, 0.0, 0.0]]).T)
    with pytest.raises(DegenerateArrayError, match="at least 3"):
        wheel_sizing(single, budget)
