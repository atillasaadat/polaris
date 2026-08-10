"""Rod authority and the B-dot noise floor (design doc §7, §8.5, §12).

Three numbers carry the whole magnetorquer chapter and each is a product of
factors that are individually easy to drop:

* the **average available torque** :math:`\\eta\\,m_{\\mathrm{in}}\\,|B|_{\\min}\\,d`
  — dropping :math:`\\eta` overstates it by 50 %, sizing at the orbit-mean field
  by another ~50 %, and using a per-rod rating instead of the inscribed radius by
  :math:`\\sqrt3` on an orthogonal triad;
* the **momentum removable in the detumble budget**, which is that torque times a
  time the assumptions own rather than the config;
* the **B-dot noise floor** :math:`\\sigma\\sqrt2/(\\Delta t|B|)`, which is a
  design *output* — it says what rate the vehicle can still tell from noise.

Each is pinned against an independent calculation, and each factor is shown
entering exactly once. :mod:`tests.analysis.test_sizing_criteria` covers the
verdicts; this covers what they are computed from.

References
----------
Sidi §7.5 [sidi1997]; Camillo & Markley [camillo1980]; Avanzini & Giulietti
[avanzini2012]; Markley & Crassidis §7.4 [markley2014].
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.sizing.assumptions import MTQ_ORBIT_EFFICIENCY, SizingAssumptions
from analysis.sizing.disturbances import disturbance_budget
from analysis.sizing.envelope import DegenerateArrayError, envelope
from analysis.sizing.magnetorquers import bdot_noise_floor, criteria, mtq_sizing
from analysis.sizing.wheels import wheel_sizing


@pytest.fixture()
def budget(vehicle):
    """The reference vehicle's disturbance budget — the field statistics source."""
    return disturbance_budget(vehicle)


def _criterion(rows, fragment):
    matches = [c for c in rows if fragment in c.name]
    assert len(matches) == 1, f"{fragment!r} matched {[c.name for c in matches]}"
    return matches[0]


# --------------------------------------------------------------------------
# Available torque: every factor, exactly once
# --------------------------------------------------------------------------


def test_the_average_torque_is_the_four_factor_product(vehicle, budget):
    """:math:`\\bar\\tau = \\eta\\,m_{\\mathrm{in}}\\,|B|_{\\min}\\,d`, factor by factor.

    Built from the rod-set envelope's own inscribed radius, the budget's own
    minimum field, the config's duty factor and the assumption's efficiency — so
    the assertion is the product identity rather than a torque in N·m that
    belongs to one rod catalog entry.
    """
    sizing = mtq_sizing(vehicle, budget)
    inscribed = envelope(vehicle.mtq_axes, vehicle.mtq_max_dipole_am2).inscribed
    assert sizing.dipole.inscribed == pytest.approx(inscribed)
    assert sizing.average_torque_nm == pytest.approx(
        MTQ_ORBIT_EFFICIENCY * inscribed * budget.field.min_t * vehicle.mtq_duty_factor
    )


def test_the_efficiency_and_the_duty_factor_are_each_applied_once(vehicle, budget):
    """``average / peak`` is exactly :math:`\\eta d` — no factor applied twice.

    The peak figure is the instantaneous torque with the dipole perpendicular to
    the field: no orbit-average projection loss, no duty cycling. Their ratio is
    therefore the two derating factors and nothing else, which is a sharper check
    than either number alone — a second application of :math:`\\eta` somewhere
    downstream would leave both plausible and the ratio wrong.
    """
    assumptions = SizingAssumptions(mtq_efficiency=0.4)
    duty = 0.25
    sizing = mtq_sizing(
        dataclasses.replace(vehicle, mtq_duty_factor=duty), budget, assumptions
    )
    assert sizing.peak_torque_nm == pytest.approx(
        sizing.dipole.inscribed * budget.field.min_t
    )
    assert sizing.average_torque_nm / sizing.peak_torque_nm == pytest.approx(0.4 * duty)
    assert sizing.average_torque_nm < sizing.peak_torque_nm


def test_authority_is_sized_at_the_weakest_field_not_the_mean(vehicle, budget):
    """:math:`|B|_{\\min}` is what the rods get judged on, and it is well below the mean.

    An authority claim made at the orbit mean is one the vehicle cannot honour
    for part of every lap. The ratio between the two is a property of the orbit
    and is not pinned; what is pinned is which end the sizing uses, and that
    substituting the mean would inflate the answer.
    """
    sizing = mtq_sizing(vehicle, budget)
    assert sizing.field_min_t == budget.field.min_t
    assert budget.field.min_t < budget.field.mean_t
    at_mean = (
        MTQ_ORBIT_EFFICIENCY
        * sizing.dipole.inscribed
        * budget.field.mean_t
        * vehicle.mtq_duty_factor
    )
    assert sizing.average_torque_nm < at_mean


def test_an_orthogonal_triad_guarantees_its_rating_and_not_its_diagonal(
    vehicle, budget
):
    """Three orthogonal rods guarantee :math:`m`, not :math:`m\\sqrt3`.

    The corner of the dipole cube reaches :math:`\\sqrt3` times the per-rod
    rating, and quoting that as capability is the classic sizing error: it is
    available in four directions out of the sphere. The inscribed radius is what
    the vehicle can produce whichever way the momentum error happens to point.
    """
    triad = dataclasses.replace(vehicle, mtq_axes=np.eye(3), mtq_max_dipole_am2=12.0)
    sizing = mtq_sizing(triad, budget)
    assert sizing.dipole.inscribed == pytest.approx(12.0)
    assert sizing.dipole.circumscribed == pytest.approx(12.0 * np.sqrt(3.0), rel=1e-3)


def test_a_rod_set_that_cannot_span_three_axes_is_refused(vehicle, budget):
    """Two rods, or three coplanar ones, raise rather than reporting an authority.

    Magnetic actuation is already rank-deficient instant by instant; a set that
    is *geometrically* deficient as well has no dipole envelope at all, and a
    number computed for one would describe an axis the vehicle can never torque
    about as merely weak.
    """
    coplanar = dataclasses.replace(
        vehicle,
        mtq_axes=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).T,
    )
    with pytest.raises(DegenerateArrayError, match="coplanar"):
        mtq_sizing(coplanar, budget)


# --------------------------------------------------------------------------
# What the authority has to accomplish
# --------------------------------------------------------------------------


def test_the_tip_off_momentum_is_the_same_quantity_the_wheels_call_d1(vehicle, budget):
    """One tip-off momentum, two modules — the rods remove what the wheels would hold.

    ``mtq.tipoff_momentum_nms`` and the wheels' D1 driver are the same
    :math:`J_{\\max}\\omega_{tipoff}`, and the whole rods-then-wheels CONOPS
    argument rests on them being the same number. Two independent computations of
    it could drift and the report would still read as though one had settled the
    other.
    """
    assumptions = SizingAssumptions(tipoff_rate_radps=np.deg2rad(3.0))
    mtq = mtq_sizing(vehicle, budget, assumptions)
    d1 = wheel_sizing(vehicle, budget, assumptions).drivers[0]
    assert mtq.tipoff_momentum_nms == pytest.approx(d1.required_nms)
    assert mtq.tipoff_momentum_nms == pytest.approx(
        float(np.max(vehicle.principal_moments_kgm2)) * np.deg2rad(3.0)
    )


def test_removable_momentum_and_implied_duration_are_each_others_inverse(
    vehicle, budget
):
    """``removable = tau*T`` and ``implied = h/tau`` — one torque, two readings.

    The criterion is written on the momentum form and the report prints the time
    form, so they must agree: the implied duration equals the budget exactly when
    the removable momentum equals the tip-off momentum. Asserting that identity
    is what stops the printed duration becoming decoration.
    """
    sizing = mtq_sizing(vehicle, budget)
    assert sizing.removable_momentum_nms == pytest.approx(
        sizing.average_torque_nm * sizing.detumble_budget_s
    )
    assert sizing.implied_detumble_s == pytest.approx(
        sizing.tipoff_momentum_nms / sizing.average_torque_nm
    )
    assert sizing.detumble_budget_s == pytest.approx(vehicle.orbit.period_s)

    ratio = sizing.removable_momentum_nms / sizing.tipoff_momentum_nms
    assert sizing.implied_detumble_s == pytest.approx(sizing.detumble_budget_s / ratio)


def test_a_zero_duty_factor_removes_nothing_and_takes_forever(vehicle, budget):
    """Rods that are never energised have no authority, and the report must say so.

    A duty factor of zero is a config defect — the MTQ/MAG interlock's quiet
    window taken to its limit — and the failure mode to avoid is a small-looking
    torque that still passes something. The implied duration goes to infinity
    rather than dividing by zero, and both magnetic criteria fail.
    """
    dead = dataclasses.replace(vehicle, mtq_duty_factor=0.0)
    sizing = mtq_sizing(dead, budget)
    assert sizing.average_torque_nm == 0.0
    assert sizing.removable_momentum_nms == 0.0
    assert sizing.implied_detumble_s == float("inf")

    rows = criteria(dead, sizing)
    assert not _criterion(rows, "M1 desaturation authority").passes
    assert not _criterion(rows, "M2 detumble authority").passes


def test_the_detumble_budget_is_an_assumption_not_a_config_value(vehicle, budget):
    """One orbit by default, and the override reaches the removable momentum.

    Detumble is field-geometry limited, so the budget is an operational
    assumption rather than something rod sizing can buy — which means it has to
    be settable, and has to move the answer linearly when it is.
    """
    default = mtq_sizing(vehicle, budget)
    half = mtq_sizing(
        vehicle,
        budget,
        SizingAssumptions(detumble_budget_s=0.5 * vehicle.orbit.period_s),
    )
    assert half.detumble_budget_s == pytest.approx(0.5 * default.detumble_budget_s)
    assert half.removable_momentum_nms == pytest.approx(
        0.5 * default.removable_momentum_nms
    )


# --------------------------------------------------------------------------
# The B-dot noise floor
# --------------------------------------------------------------------------


def test_the_noise_floor_is_the_differenced_field_noise_over_the_field(vehicle, budget):
    """:math:`\\omega_{floor} = \\sigma\\sqrt2/(\\Delta t|B|)`, computed independently.

    Two samples each carrying :math:`\\sigma` give a difference of variance
    :math:`2\\sigma^2` — the :math:`\\sqrt2` is the part that is easy to lose, and
    losing it understates the floor by 41 % in the direction that lets a too-low
    exit threshold pass. The expectation here is formed from the vehicle's own
    magnetometer noise and control period, with the :math:`\\sqrt2` written out.
    """
    floor = bdot_noise_floor(vehicle, budget)
    derivative_noise = vehicle.mag_noise_t * np.sqrt(2.0) / vehicle.control_period_s
    assert floor.derivative_noise_t_s == pytest.approx(derivative_noise)
    assert floor.rate_worst_radps == pytest.approx(
        derivative_noise / budget.field.min_t
    )
    assert floor.rate_mean_radps == pytest.approx(
        derivative_noise / budget.field.mean_t
    )
    assert floor.sigma_t == vehicle.mag_noise_t
    assert floor.sample_period_s == vehicle.control_period_s


def test_the_floor_is_worst_where_the_field_is_weakest(vehicle, budget):
    """The two reported floors differ by exactly the field ratio.

    :math:`|\\mathrm dB/\\mathrm dt| \\approx |\\omega\\times B|`, so a weaker
    field produces a weaker signal against the same measurement noise. The
    bounding figure is the one at :math:`|B|_{\\min}` and the mean is reported
    beside it; their ratio being exactly :math:`|B|_{\\max\\text{-ish}}/|B|_{\\min}`
    is what shows the same noise entered both.
    """
    floor = bdot_noise_floor(vehicle, budget)
    assert floor.rate_worst_radps > floor.rate_mean_radps
    assert floor.rate_worst_radps / floor.rate_mean_radps == pytest.approx(
        budget.field.mean_t / budget.field.min_t
    )


def test_a_longer_differencing_interval_lowers_the_floor_proportionally(
    vehicle, budget
):
    """The design lever, stated as a scaling: :math:`\\omega_{floor}\\propto1/\\Delta t`.

    When the floor is the binding constraint, differencing over a longer interval
    is the cheapest way out — bounded above by ``BdotMaxSampleDtSec``, past which
    the secant stops approximating the tangent. The linearity is what makes that
    trade computable from the report instead of by experiment.
    """
    base = bdot_noise_floor(vehicle, budget)
    slower = bdot_noise_floor(
        dataclasses.replace(vehicle, control_period_s=4.0 * vehicle.control_period_s),
        budget,
    )
    quieter = bdot_noise_floor(
        dataclasses.replace(vehicle, mag_noise_t=0.25 * vehicle.mag_noise_t), budget
    )
    assert slower.rate_worst_radps == pytest.approx(0.25 * base.rate_worst_radps)
    assert quieter.rate_worst_radps == pytest.approx(0.25 * base.rate_worst_radps)


# --------------------------------------------------------------------------
# The three criteria the numbers above are judged by
# --------------------------------------------------------------------------


def test_the_three_criteria_carry_the_margin_on_the_quantity_they_judge(
    vehicle, budget
):
    """M1 against secular torque, M2 against tip-off momentum, M3 against the floor.

    Each threshold is ``margin * demand`` on the quantity named in the row, in
    the row's own units — M3 in deg/s on both sides, the two momentum and torque
    rows in SI. A unit slipping on one side of a comparison is a factor of 57 and
    reads as a comfortable pass.
    """
    assumptions = SizingAssumptions(margin=1.5)
    sizing = mtq_sizing(vehicle, budget, assumptions)
    rows = criteria(vehicle, sizing, assumptions)

    m1 = _criterion(rows, "M1 desaturation authority")
    assert m1.threshold == pytest.approx(1.5 * budget.secular_nm)
    assert m1.measured == pytest.approx(sizing.average_torque_nm)
    assert m1.units == "N.m" and m1.sense == "min"

    m2 = _criterion(rows, "M2 detumble authority")
    assert m2.threshold == pytest.approx(1.5 * sizing.tipoff_momentum_nms)
    assert m2.measured == pytest.approx(sizing.removable_momentum_nms)

    m3 = _criterion(rows, "M3 DetumbleExitRadps")
    assert m3.units == "deg/s"
    assert m3.threshold == pytest.approx(
        np.degrees(sizing.noise_floor.rate_worst_radps)
    )
    assert m3.measured == pytest.approx(np.degrees(vehicle.detumble_exit_radps))


def test_m1_is_written_on_the_secular_half_of_the_budget_only(vehicle):
    """The cyclic half sizes wheels, not rods — moving torque between them shows it.

    A cyclic torque gives its momentum back within a lap, so no unloading
    authority is needed for it; the secular half is what accumulates without
    bound. Reassigning the same total from cyclic to secular must raise M1's
    threshold and nothing about the rods' capability.
    """
    cyclic = disturbance_budget(vehicle, SizingAssumptions(secular_fraction_srp=0.0))
    secular = disturbance_budget(
        vehicle,
        SizingAssumptions(
            secular_fraction_gg=1.0,
            secular_fraction_aero=1.0,
            secular_fraction_srp=1.0,
            secular_fraction_mag=1.0,
        ),
    )
    lenient = mtq_sizing(vehicle, cyclic)
    strict = mtq_sizing(vehicle, secular)
    assert strict.secular_torque_nm > lenient.secular_torque_nm
    assert strict.average_torque_nm == pytest.approx(lenient.average_torque_nm)
    assert (
        _criterion(criteria(vehicle, strict), "M1 desaturation").threshold
        > _criterion(criteria(vehicle, lenient), "M1 desaturation").threshold
    )
