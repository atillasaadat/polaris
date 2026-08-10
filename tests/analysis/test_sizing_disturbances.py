"""The §5.3 disturbance-torque budget, term by term (design doc §5.3, §12).

Four closed forms feed everything downstream: the cyclic half sizes wheel
storage (D2), the secular half sizes the rods (M1), and an error in either is an
error in every margin the report prints. So each term is checked twice — once
against a **hand-computable case** whose arithmetic is written out in the test,
and once against its **scaling law**, which is vehicle-independent and cannot rot
when the reference bus is re-baselined:

============================  ====================================
term                          scaling
============================  ====================================
gravity gradient              :math:`\\propto \\Delta I / R^3`
aerodynamic                   :math:`\\propto \\rho V^2 = \\rho\\mu/a`
solar radiation pressure      independent of orbit radius
residual magnetic             :math:`\\propto |m|\\,|B|`
============================  ====================================

Vehicles are built with :func:`dataclasses.replace` off the committed fixture so
the numbers under test are chosen here rather than inherited, and no assertion
records a measurement of whichever bus happens to be current.

References
----------
Wertz §17 [wertz1978]; Wertz, Everett & Puschell §19.2 [wertz2011]; Vallado
§8.6.2 and Table 8-4 [vallado2013]; Montenbruck & Gill §3.4 [montenbruck2000].
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.disturbances import (
    EARTH_GM_M3_S2,
    SOLAR_CONSTANT_W_M2,
    SPEED_OF_LIGHT_M_S,
    aerodynamic_torque,
    disturbance_budget,
    exponential_density,
    field_statistics,
    gravity_gradient_torque,
    residual_dipole_torque,
    srp_torque,
)
from analysis.sizing.magnetorquers import mtq_sizing


def _at(vehicle, **fields):
    """The fixture vehicle with @p fields replaced — a chosen vehicle, not the bus."""
    orbit_fields = {k: fields.pop(k) for k in ("sma_m",) if k in fields}
    if orbit_fields:
        fields["orbit"] = dataclasses.replace(vehicle.orbit, **orbit_fields)
    return dataclasses.replace(vehicle, **fields)


def _term(budget, name):
    """The one budget term called @p name."""
    return next(t for t in budget.terms if t.name == name)


# --------------------------------------------------------------------------
# The constants the closed forms are built on
# --------------------------------------------------------------------------


def test_the_radiation_pressure_constant_is_the_textbook_value():
    """:math:`\\Phi/c = 4.54\\,\\mu`Pa at 1 AU — the number every SRP torque scales.

    A wrong solar constant or a wrong :math:`c` would scale every SRP result by a
    factor no downstream test could distinguish from a lever-arm change, so the
    ratio is pinned against the published value ([montenbruck2000] §3.4 with the
    modern TSI of 1361 W/m²) rather than only against itself.
    """
    assert SOLAR_CONSTANT_W_M2 / SPEED_OF_LIGHT_M_S == pytest.approx(4.54e-6, rel=1e-3)


def test_the_gravitational_parameter_agrees_with_the_orbit_model(vehicle):
    """One :math:`\\mu` across the packages: :math:`n^2 a^3 = \\mu`.

    The gravity-gradient term uses :data:`EARTH_GM_M3_S2` directly while
    :class:`analysis.control.vehicle.Orbit` computes mean motion from its own
    copy. Two constants that drift apart would put the disturbance budget and the
    orbit period on different Earths, which is exactly the kind of disagreement
    no single-module test can see.
    """
    n = vehicle.orbit.mean_motion_rad_s
    assert n**2 * vehicle.orbit.sma_m**3 == pytest.approx(EARTH_GM_M3_S2, rel=1e-12)


# --------------------------------------------------------------------------
# Gravity gradient
# --------------------------------------------------------------------------


def test_gravity_gradient_matches_a_hand_computed_case(vehicle):
    """:math:`\\tfrac{3\\mu}{2R^3}\\Delta I` at :math:`R = 7000` km, :math:`\\Delta I = 6`.

    Worked by hand: :math:`1.5 \\times 3.986004418\\times10^{14} \\times 6 =
    3.58740\\times10^{15}`, over :math:`R^3 = 3.43\\times10^{20}` m³, giving
    :math:`1.04589\\times10^{-5}` N·m. Both the radius and the inertia spread are
    chosen here, so the expected value is arithmetic a reviewer can redo rather
    than a figure read off a run.
    """
    subject = _at(vehicle, inertia_kgm2=np.diag([10.0, 7.0, 4.0]), sma_m=7.0e6)
    assert gravity_gradient_torque(subject) == pytest.approx(1.04589e-5, rel=1e-5)


def test_gravity_gradient_scales_as_one_over_r_cubed_and_linearly_in_the_spread(
    vehicle,
):
    """:math:`\\tau_{gg} \\propto \\Delta I/R^3`, checked as two ratios.

    The exponent on :math:`R` is the part that is easy to get wrong and
    impossible to see in a single magnitude — a :math:`1/R^2` slip is a factor of
    1.1 at LEO and a factor of 6 at GEO. Doubling the radius must divide the
    torque by exactly eight, and tripling the spread must triple it, on any
    vehicle.
    """
    base = _at(vehicle, inertia_kgm2=np.diag([10.0, 7.0, 4.0]), sma_m=7.0e6)
    higher = _at(base, sma_m=1.4e7)
    fatter = _at(base, inertia_kgm2=np.diag([22.0, 7.0, 4.0]))  # spread 6 -> 18
    assert gravity_gradient_torque(higher) == pytest.approx(
        gravity_gradient_torque(base) / 8.0
    )
    assert gravity_gradient_torque(fatter) == pytest.approx(
        3.0 * gravity_gradient_torque(base)
    )


def test_a_spherical_inertia_produces_no_gravity_gradient_torque(vehicle):
    """:math:`\\Delta I = 0` is the analytic zero of the term, not a small number.

    The couple exists because the mass distribution is not isotropic; a body with
    equal principal moments has none at any altitude or attitude. A formula that
    returned something here would be carrying a stray additive term.
    """
    ball = _at(vehicle, inertia_kgm2=np.diag([9.0, 9.0, 9.0]))
    assert gravity_gradient_torque(ball) == 0.0


# --------------------------------------------------------------------------
# Aerodynamic
# --------------------------------------------------------------------------


def test_aerodynamic_torque_matches_a_hand_computed_case(vehicle):
    """:math:`\\tfrac12\\rho V^2 C_d A |d|` with the speed derived a second way.

    At :math:`a = 7000` km the circular speed is :math:`V = na = 7546.0` m/s, so
    with :math:`\\rho = 10^{-12}` kg/m³, :math:`C_d = 2`, :math:`A = 1` m² and a
    0.1 m lever arm the torque is :math:`0.5\\times10^{-12}\\times7546^2\\times0.2
    = 5.694\\times10^{-6}` N·m. The expected value is formed from the **mean
    motion** (:math:`V = na`) rather than from :math:`\\sqrt{\\mu/a}`, so the two
    routes to the orbital speed have to agree as well.
    """
    subject = _at(
        vehicle,
        sma_m=7.0e6,
        drag_cd=2.0,
        drag_area_m2=1.0,
        cp_offset_aero_m=np.array([0.1, 0.0, 0.0]),
    )
    speed = subject.orbit.mean_motion_rad_s * subject.orbit.sma_m
    expected = 0.5 * 1.0e-12 * speed**2 * 2.0 * 1.0 * 0.1
    assert aerodynamic_torque(subject, 1.0e-12) == pytest.approx(expected, rel=1e-9)
    assert aerodynamic_torque(subject, 1.0e-12) == pytest.approx(5.694e-6, rel=1e-3)


def test_aerodynamic_torque_scales_as_rho_v_squared(vehicle):
    """Linear in density, and :math:`V^2 = \\mu/a` makes it inverse in the radius.

    The dynamic pressure is the whole altitude dependence of this term: density
    falls exponentially and :math:`V^2` falls as :math:`1/a`. Doubling the
    semi-major axis at fixed density must halve the torque exactly — a check that
    holds for any vehicle and catches a :math:`V` used where :math:`V^2` belongs.
    """
    base = _at(vehicle, sma_m=7.0e6)
    higher = _at(vehicle, sma_m=1.4e7)
    assert aerodynamic_torque(base, 2.0e-12) == pytest.approx(
        2.0 * aerodynamic_torque(base, 1.0e-12)
    )
    assert aerodynamic_torque(higher, 1.0e-12) == pytest.approx(
        0.5 * aerodynamic_torque(base, 1.0e-12)
    )


def test_aerodynamic_torque_is_linear_in_the_plate_and_in_the_lever_arm(vehicle):
    """:math:`C_d`, :math:`A` and :math:`|d_{cp}|` each enter to the first power.

    And only the *magnitude* of the lever arm enters, which is what makes the
    figure a worst case over attitude: rotating the offset must not change it.
    """
    base = _at(
        vehicle,
        drag_cd=2.0,
        drag_area_m2=1.0,
        cp_offset_aero_m=np.array([0.0, 0.0, 0.05]),
    )
    torque = aerodynamic_torque(base, 1.0e-12)
    assert aerodynamic_torque(_at(base, drag_cd=4.0), 1.0e-12) == pytest.approx(
        2.0 * torque
    )
    assert aerodynamic_torque(_at(base, drag_area_m2=3.0), 1.0e-12) == pytest.approx(
        3.0 * torque
    )
    rotated = _at(base, cp_offset_aero_m=np.array([0.03, 0.04, 0.0]))
    assert aerodynamic_torque(rotated, 1.0e-12) == pytest.approx(torque)


def test_a_vehicle_with_no_lever_arm_has_no_surface_torques(vehicle):
    """CP on CM means no aerodynamic and no SRP couple — the analytic zero.

    Worth pinning because the config declares these offsets with no default
    precisely so that "measured as zero" and "never measured" cannot look alike;
    if the formulas carried a floor, that distinction would be invisible.
    """
    balanced = _at(
        vehicle,
        cp_offset_aero_m=np.zeros(3),
        cp_offset_srp_m=np.zeros(3),
    )
    assert aerodynamic_torque(balanced, 1.0e-12) == 0.0
    assert srp_torque(balanced) == 0.0


# --------------------------------------------------------------------------
# Solar radiation pressure
# --------------------------------------------------------------------------


def test_srp_matches_a_hand_computed_case(vehicle):
    """:math:`(\\Phi/c)AC_r|d|` with :math:`A C_r |d| = 100` m³ by construction.

    :math:`A = 10` m², :math:`C_r = 2`, :math:`|d| = |(3,4,0)| = 5` m, so the
    torque is exactly one hundred times the radiation pressure:
    :math:`100 \\times 4.5398\\times10^{-6} = 4.5398\\times10^{-4}` N·m. Choosing
    a round product is what makes this checkable without re-running the code.
    """
    subject = _at(
        vehicle,
        srp_area_m2=10.0,
        srp_cr=2.0,
        cp_offset_srp_m=np.array([3.0, 4.0, 0.0]),
    )
    assert srp_torque(subject) == pytest.approx(4.5398e-4, rel=1e-4)


def test_srp_does_not_depend_on_the_orbit_radius(vehicle):
    """The Sun is 1 AU away from LEO and from GEO alike.

    The solar constant is evaluated at 1 AU and the few thousand kilometres of
    orbit radius change it by parts in :math:`10^5`, which the model neglects
    deliberately. The invariant is exact here, and it is the one that separates
    SRP from every other term in the budget: it is the disturbance that does not
    go away by flying higher.
    """
    low = _at(vehicle, sma_m=6.9e6)
    high = _at(vehicle, sma_m=4.2e7)
    assert srp_torque(high) == pytest.approx(srp_torque(low), rel=1e-12)


def test_srp_is_linear_in_area_reflectivity_and_lever_arm(vehicle):
    """Each of :math:`A`, :math:`C_r` and :math:`|d|` enters to the first power.

    :math:`C_r` is :math:`1+q`: 1 for a perfect absorber, 2 for a perfect
    specular reflector, so a black vehicle sees half the torque of a mirrored one
    with the same geometry.
    """
    absorber = _at(
        vehicle, srp_area_m2=2.0, srp_cr=1.0, cp_offset_srp_m=np.array([0.2, 0.0, 0.0])
    )
    base = srp_torque(absorber)
    assert srp_torque(_at(absorber, srp_cr=2.0)) == pytest.approx(2.0 * base)
    assert srp_torque(_at(absorber, srp_area_m2=5.0)) == pytest.approx(2.5 * base)
    assert srp_torque(
        _at(absorber, cp_offset_srp_m=np.array([0.0, 0.6, 0.0]))
    ) == pytest.approx(3.0 * base)


# --------------------------------------------------------------------------
# Residual magnetic
# --------------------------------------------------------------------------


def test_the_residual_dipole_torque_is_bilinear_in_dipole_and_field(vehicle):
    """:math:`|m||B|`: double either factor and the torque doubles.

    Only the magnitudes enter, so this is the worst case over the angle between
    them — the :math:`\\sin` of that angle is taken as one, which is the right
    treatment for a sizing bound and the reason the term never vanishes.
    """
    subject = _at(vehicle, residual_dipole_am2=np.array([0.03, 0.04, 0.0]))
    assert residual_dipole_torque(subject, 5.0e-5) == pytest.approx(0.05 * 5.0e-5)
    assert residual_dipole_torque(subject, 1.0e-4) == pytest.approx(
        2.0 * residual_dipole_torque(subject, 5.0e-5)
    )
    doubled = _at(vehicle, residual_dipole_am2=np.array([0.06, 0.08, 0.0]))
    assert residual_dipole_torque(doubled, 5.0e-5) == pytest.approx(
        2.0 * residual_dipole_torque(subject, 5.0e-5)
    )


# --------------------------------------------------------------------------
# The atmosphere the aerodynamic term is evaluated in
# --------------------------------------------------------------------------


def test_the_density_is_log_linear_inside_a_band():
    """:math:`\\rho\\propto e^{-h/H}` means the geometric mean rule holds exactly.

    For three equally spaced altitudes inside one band,
    :math:`\\rho_2^2 = \\rho_1\\rho_3`. That is the exponential law itself, and it
    is checked without naming a scale height so the test survives a Table 8-4
    update in ``sim/world/atmosphere.cpp``. The 510-550 km points sit inside a
    single band, which the equality itself confirms — it would break across a
    band boundary.
    """
    low, mid, high = (exponential_density(h) for h in (510e3, 530e3, 550e3))
    assert mid**2 == pytest.approx(low * high, rel=1e-12)


def test_the_density_falls_monotonically_across_every_band(vehicle):
    """No altitude in the flown range is denser than one below it.

    The table is piecewise, and a mis-sorted or mis-parsed band would show up as
    a step *upwards* somewhere in the middle of the range — plausible-looking in
    any single evaluation and wrong in the direction that lowers the aerodynamic
    torque.
    """
    altitudes = np.arange(100e3, 1000e3, 10e3)
    densities = np.array([exponential_density(h) for h in altitudes])
    assert np.all(np.diff(densities) < 0.0)


def test_a_source_without_a_band_table_is_refused(tmp_path):
    """A missing table raises rather than defaulting to a plausible density.

    A silently defaulted density gives an aerodynamic torque with no provenance,
    which is worse than no answer: it looks like a measurement of the vehicle's
    environment and is a measurement of nothing.
    """
    empty = tmp_path / "atmosphere.cpp"
    empty.write_text("// no table here\nint main() { return 0; }\n")
    with pytest.raises(ValueError, match="no exponential-atmosphere band table"):
        exponential_density(500e3, source=empty)


# --------------------------------------------------------------------------
# The assembled budget
# --------------------------------------------------------------------------


def test_every_budget_term_is_its_own_closed_form(vehicle):
    """The four terms are the four functions, evaluated at the budget's own inputs.

    This is what stops the assembly drifting from the formulas the tests above
    pin: the budget must be exactly ``gravity_gradient_torque``,
    ``aerodynamic_torque`` at the parsed density, ``srp_torque`` and
    ``residual_dipole_torque`` at the orbit field — no rescaling, no second copy
    of a coefficient.
    """
    budget = disturbance_budget(vehicle)
    assert _term(budget, "gravity gradient").torque_nm == pytest.approx(
        gravity_gradient_torque(vehicle)
    )
    assert _term(budget, "aerodynamic").torque_nm == pytest.approx(
        aerodynamic_torque(vehicle, budget.density_kg_m3)
    )
    assert _term(budget, "solar radiation pressure").torque_nm == pytest.approx(
        srp_torque(vehicle)
    )
    assert budget.density_kg_m3 == pytest.approx(exponential_density(budget.altitude_m))
    assert budget.altitude_m == pytest.approx(vehicle.orbit.sma_m - 6378137.0)


def test_the_magnetic_term_is_taken_at_the_strongest_field_the_rods_at_the_weakest(
    vehicle,
):
    """One field model, two ends of it, and each used where it is conservative.

    A disturbance is worst where :math:`|B|` is largest; magnetic *authority* is
    worst where it is smallest. Both come from the same
    :func:`field_statistics` call on the same orbit, so this pins the pairing
    across the two modules — the place where taking the mean, or taking the same
    end twice, would flatter the design at both ends at once.
    """
    budget = disturbance_budget(vehicle)
    magnitude = float(np.linalg.norm(vehicle.residual_dipole_am2))
    assert _term(budget, "residual magnetic").torque_nm == pytest.approx(
        magnitude * budget.field.max_t
    )
    assert mtq_sizing(vehicle, budget).field_min_t == budget.field.min_t
    assert budget.field.min_t < budget.field.max_t


def test_each_term_splits_into_the_secular_and_cyclic_halves_it_was_assigned(vehicle):
    """``secular + cyclic == total`` per term, at the fraction the assumption states.

    The split decides which actuator each term sizes — cyclic torque sizes wheel
    storage, secular torque sizes the rods — so an assumption that did not reach
    the term it names would move a driver onto the wrong actuator silently.
    """
    assumed = SizingAssumptions(
        secular_fraction_gg=0.25,
        secular_fraction_aero=0.5,
        secular_fraction_srp=0.75,
        secular_fraction_mag=1.0,
    )
    budget = disturbance_budget(vehicle, assumed)
    fractions = {
        "gravity gradient": 0.25,
        "aerodynamic": 0.5,
        "solar radiation pressure": 0.75,
        "residual magnetic": 1.0,
    }
    for term in budget.terms:
        assert term.secular_fraction == fractions[term.name]
        assert term.secular_nm == pytest.approx(fractions[term.name] * term.torque_nm)
        assert term.secular_nm + term.cyclic_nm == pytest.approx(term.torque_nm)
    assert budget.secular_nm == pytest.approx(sum(t.secular_nm for t in budget.terms))


def test_the_total_is_a_sum_and_not_a_root_sum_of_squares(vehicle):
    """Deterministic torques whose worst cases can coincide are added, not RSS'd.

    RSS understates a budget with two comparable terms by about 30 %, and these
    are not independent random variables — the sizing case is the coincidence.
    The two aggregations differ by enough on the reference vehicle that the test
    distinguishes them without pinning either.
    """
    budget = disturbance_budget(vehicle)
    torques = np.array([t.torque_nm for t in budget.terms])
    assert budget.total_nm == pytest.approx(float(np.sum(torques)))
    assert budget.total_nm > float(np.linalg.norm(torques))


def test_refining_the_field_sampling_can_only_tighten_the_bracket(vehicle):
    """A nested, denser sample of the same orbit cannot report a weaker extreme.

    ``linspace(0, T, 721)`` is a subset of ``linspace(0, T, 1441)``, so the
    denser run sees every point the coarser one did: its minimum can only fall
    and its maximum can only rise. That is the sampling-convergence claim the
    default of 721 rests on, and it holds without pinning a field magnitude that
    belongs to the orbit rather than to this package.
    """
    coarse = field_statistics(vehicle, samples=721)
    fine = field_statistics(vehicle, samples=1441)
    assert fine.min_t <= coarse.min_t + 1e-18
    assert fine.max_t >= coarse.max_t - 1e-18
    assert fine.samples == 1441
    assert coarse.min_t <= coarse.mean_t <= coarse.max_t
