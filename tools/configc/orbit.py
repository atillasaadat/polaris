"""Keplerian-element to ECI Cartesian conversion for the config compiler (§19.3).

The compiler resolves; consumers do not re-derive. The human-edited config states
an orbit as classical elements because that is what a mission designer writes,
but the sim propagator wants an ECI state vector — so the conversion happens here,
once, and ``sim_setup.json`` carries the Cartesian state. No C++ orbital-element
code is needed downstream.

Reference
---------
Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed.,
Microcosm Press, 2013, §2.6 — classical orbital elements to position and
velocity (the "COE2RV" transformation).
"""

from __future__ import annotations

import math

#: Geocentric gravitational constant [m³/s²]. Must equal ``wgs84::kGM`` in
#: lib/constants/constants.hpp so the Python-side conversion and the C++ sim
#: agree bit-for-bit on the same orbit.
MU_EARTH = 3.986004418e14

Vec3 = tuple[float, float, float]


def keplerian_to_cartesian(
    sma_m: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu: float = MU_EARTH,
) -> tuple[Vec3, Vec3]:
    """Convert classical orbital elements to an ECI position and velocity.

    The state is built in the perifocal (PQW) frame and rotated into ECI by
    ``Rz(-raan) Rx(-inc) Rz(-argp)``.

    Parameters
    ----------
    sma_m : float
        Semi-major axis [m], must be positive.
    ecc : float
        Eccentricity, ``0 <= ecc < 1`` (closed orbits only).
    inc_deg : float
        Inclination [deg].
    raan_deg : float
        Right ascension of the ascending node [deg].
    argp_deg : float
        Argument of periapsis [deg].
    true_anomaly_deg : float
        True anomaly [deg].
    mu : float, optional
        Gravitational parameter [m³/s²]; defaults to :data:`MU_EARTH`.

    Returns
    -------
    position_m : tuple of float
        ECI position [m].
    velocity_m_s : tuple of float
        ECI velocity [m/s].

    Raises
    ------
    ValueError
        If ``sma_m <= 0`` or ``ecc`` is outside ``[0, 1)``.
    """
    if sma_m <= 0.0:
        raise ValueError(f"sma_m must be positive, got {sma_m}")
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"ecc must satisfy 0 <= ecc < 1, got {ecc}")

    inc = math.radians(inc_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)
    nu = math.radians(true_anomaly_deg)

    p = sma_m * (1.0 - ecc * ecc)  # semi-latus rectum [m]
    r = p / (1.0 + ecc * math.cos(nu))
    sqrt_mu_p = math.sqrt(mu / p)

    r_pqw = (r * math.cos(nu), r * math.sin(nu), 0.0)
    v_pqw = (-sqrt_mu_p * math.sin(nu), sqrt_mu_p * (ecc + math.cos(nu)), 0.0)

    ci, si = math.cos(inc), math.sin(inc)
    cr, sr = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    # PQW -> ECI rotation, row-major.
    rot = (
        (cr * cw - sr * sw * ci, -cr * sw - sr * cw * ci, sr * si),
        (sr * cw + cr * sw * ci, -sr * sw + cr * cw * ci, -cr * si),
        (sw * si, cw * si, ci),
    )
    return _apply(rot, r_pqw), _apply(rot, v_pqw)


def _apply(rot: tuple[Vec3, Vec3, Vec3], v: Vec3) -> Vec3:
    return (
        rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
        rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
        rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
    )
