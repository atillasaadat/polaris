"""Tests for the Keplerian->Cartesian conversion in the config compiler (§19.3).

Checks the transformation against closed-form invariants (speed, radius, angular
momentum, periapsis/apoapsis geometry) and round-trips through an independent
inverse implemented here as the test's own oracle.
"""

from __future__ import annotations

import math

import pytest

from configc.orbit import MU_EARTH, keplerian_to_cartesian

_SMA = 6_878_137.0  # 500 km circular, matches the LEO template config [m]


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _cartesian_to_keplerian(r, v, mu=MU_EARTH):
    """Independent inverse (RV2COE) used only as this test's round-trip oracle.

    Returns ``(sma_m, ecc, inc_deg, raan_deg, argp_deg, true_anomaly_deg)``.
    Assumes an inclined elliptical orbit (the round-trip case exercised here).
    """
    r_mag, v_mag = _norm(r), _norm(v)
    h = _cross(r, v)
    n = (-h[1], h[0], 0.0)  # node vector, k x h
    e_vec = tuple(
        ((v_mag * v_mag - mu / r_mag) * r[i] - _dot(r, v) * v[i]) / mu for i in range(3)
    )
    ecc = _norm(e_vec)
    sma = 1.0 / (2.0 / r_mag - v_mag * v_mag / mu)
    inc = math.acos(h[2] / _norm(h))
    raan = math.atan2(n[1], n[0]) % (2.0 * math.pi)
    argp = math.acos(_dot(n, e_vec) / (_norm(n) * ecc))
    if e_vec[2] < 0.0:
        argp = 2.0 * math.pi - argp
    nu = math.acos(_dot(e_vec, r) / (ecc * r_mag))
    if _dot(r, v) < 0.0:
        nu = 2.0 * math.pi - nu
    return (
        sma,
        ecc,
        math.degrees(inc),
        math.degrees(raan),
        math.degrees(argp),
        math.degrees(nu),
    )


@pytest.mark.verifies("REQ-CFG-001")
def test_circular_equatorial_orbit_has_expected_radius_and_speed():
    r, v = keplerian_to_cartesian(_SMA, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert _norm(r) == pytest.approx(_SMA, rel=1e-12)
    assert _norm(v) == pytest.approx(math.sqrt(MU_EARTH / _SMA), rel=1e-12)
    # Circular orbit: the radius vector is always perpendicular to the velocity.
    assert _dot(r, v) == pytest.approx(0.0, abs=1e-6)
    assert r[2] == pytest.approx(0.0, abs=1e-9)
    assert v[2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.verifies("REQ-CFG-001")
def test_inclination_produces_expected_out_of_plane_component():
    # At the ascending node (argp = nu = 0, raan = 0) the position is along the
    # line of nodes, so all of the inclination shows up in the velocity.
    inc_deg = 45.0
    r, v = keplerian_to_cartesian(_SMA, 0.0, inc_deg, 0.0, 0.0, 0.0)
    assert r == pytest.approx((_SMA, 0.0, 0.0), abs=1e-6)
    speed = math.sqrt(MU_EARTH / _SMA)
    assert v[2] == pytest.approx(speed * math.sin(math.radians(inc_deg)), rel=1e-12)
    assert v[1] == pytest.approx(speed * math.cos(math.radians(inc_deg)), rel=1e-12)


@pytest.mark.verifies("REQ-CFG-001")
def test_eccentric_orbit_periapsis_and_apoapsis():
    ecc = 0.2
    r_peri, v_peri = keplerian_to_cartesian(_SMA, ecc, 30.0, 40.0, 50.0, 0.0)
    r_apo, v_apo = keplerian_to_cartesian(_SMA, ecc, 30.0, 40.0, 50.0, 180.0)
    assert _norm(r_peri) == pytest.approx(_SMA * (1.0 - ecc), rel=1e-12)
    assert _norm(r_apo) == pytest.approx(_SMA * (1.0 + ecc), rel=1e-12)
    # Speed is maximum at periapsis, minimum at apoapsis (vis-viva).
    assert _norm(v_peri) > _norm(v_apo)
    assert _norm(v_peri) == pytest.approx(
        math.sqrt(MU_EARTH / _SMA * (1.0 + ecc) / (1.0 - ecc)), rel=1e-12
    )


@pytest.mark.verifies("REQ-CFG-001")
@pytest.mark.parametrize("ecc", [0.0, 0.01, 0.3])
@pytest.mark.parametrize("nu_deg", [0.0, 73.0, 200.0])
def test_specific_angular_momentum_matches_closed_form(ecc, nu_deg):
    r, v = keplerian_to_cartesian(_SMA, ecc, 97.4, 120.0, 60.0, nu_deg)
    expected = math.sqrt(MU_EARTH * _SMA * (1.0 - ecc * ecc))
    assert _norm(_cross(r, v)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.verifies("REQ-CFG-001")
def test_round_trip_recovers_the_input_elements():
    elements = (_SMA, 0.15, 97.4018, 120.0, 60.0, 200.0)
    r, v = keplerian_to_cartesian(*elements)
    recovered = _cartesian_to_keplerian(r, v)
    assert recovered[0] == pytest.approx(elements[0], rel=1e-9)
    for got, want in zip(recovered[1:], elements[1:], strict=True):
        assert got == pytest.approx(want, abs=1e-8)


@pytest.mark.verifies("REQ-CFG-001")
@pytest.mark.parametrize(
    ("sma", "ecc"), [(-1.0, 0.0), (0.0, 0.0), (_SMA, 1.0), (_SMA, -0.1)]
)
def test_rejects_non_elliptical_inputs(sma, ecc):
    with pytest.raises(ValueError):
        keplerian_to_cartesian(sma, ecc, 0.0, 0.0, 0.0, 0.0)
