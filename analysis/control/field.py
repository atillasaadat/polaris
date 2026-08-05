"""Tilted-dipole geomagnetic field along the reference orbit, for analysis only.

Why a dipole and not the onboard IGRF
-------------------------------------
The magnetorquer controllability study (:mod:`analysis.control.controllability`)
needs only the **direction** the field sweeps over an orbit: what makes an
instantaneously rank-2 actuator controllable on average is that :math:`\\hat{B}`
turns. The degree-1 (tilted dipole) truncation reproduces that sweep to a few
degrees, and Polaris already has a validated full IGRF-14 implementation in
``lib/environment/igrf`` for anything that needs magnitude accuracy — it is
simply not reachable from Python until ``bindings/`` exists, and this module
notes that rather than duplicating it.

The degree-1 coefficients are **parsed from the committed IAGA table**
(``tests/golden/igrf14coeffs.txt``, the upstream file verbatim per §3.7), not
transcribed: a restated catalog value is a copy, and every copy is a place for
the analysis and the vehicle to disagree.

Model
-----
From the Gauss potential :math:`V = a(a/r)^2[g_1^0\\cos\\theta + (g_1^1\\cos\\phi
+ h_1^1\\sin\\phi)\\sin\\theta]`, with
:math:`\\vec m = (g_1^1, h_1^1, g_1^0)` in the Earth-fixed frame,

.. math::

   \\vec B(\\vec r) = \\left(\\frac{a}{r}\\right)^3
       \\left[3(\\vec m\\cdot\\hat r)\\hat r - \\vec m\\right],

with :math:`a = 6371.2` km the IGRF reference radius. The dipole is fixed in
the Earth-fixed frame and carried into the inertial frame by the Earth-rotation
angle, so both the orbital motion and the diurnal tilt rotation appear.

Assumptions, stated
-------------------
* Circular orbit at the config's semi-major axis, inclination and RAAN;
  argument of latitude advances at the Keplerian mean motion. Secular
  perturbations are irrelevant over the single orbit the Gramian integrates.
* The Earth-fixed→inertial rotation is a simple rotation about +Z at the mean
  sidereal rate; precession/nutation/polar motion are far below the accuracy
  this study needs (§3.1 has the real chain, in C++).
* The body frame is taken **coincident with the inertial frame**, consistent
  with linearising about a fixed inertial attitude. A vehicle holding an LVLH
  attitude sees the same field sequence through a known rotation, which changes
  the numbers in the Gramian but not the rank conclusion.

Units and frames
----------------
Positions [m] and fields [T] in an Earth-centred inertial frame; angles [rad];
time [s] from the orbit epoch. SI throughout.

References
----------
Alken et al., "International Geomagnetic Reference Field: the thirteenth
generation," *Earth, Planets and Space*, 2021 [alken2021] — the coefficient
set and its conventions.
Langel, "The Main Field," in *Geomagnetism* Vol. 1 [langel1987] — the spherical
harmonic potential and the dipole truncation.
Design doc §3.7 (external reference data), §5.2 (environment models).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: IGRF reference radius [m] (design doc §5.2; the value the IAGA table's
#: expansion is defined about).
IGRF_REFERENCE_RADIUS_M = 6371200.0

#: Mean sidereal rotation rate of the Earth [rad/s] (IERS conventions).
EARTH_ROTATION_RAD_S = 7.292115e-5

#: The committed IAGA coefficient table, relative to the repository root.
IGRF_TABLE = Path("tests/golden/igrf14coeffs.txt")


@dataclass(frozen=True)
class Dipole:
    """Degree-1 geomagnetic field, Earth-fixed.

    Attributes
    ----------
    moment_nt : numpy.ndarray
        :math:`(g_1^1, h_1^1, g_1^0)` [nT], shape ``(3,)`` — the dipole vector
        in the Earth-fixed frame, in the units the IAGA table publishes.
    epoch_year : float
        Decimal year of the coefficient column used.
    """

    moment_nt: np.ndarray
    epoch_year: float

    def field_ecef_t(self, position_ecef_m: np.ndarray) -> np.ndarray:
        """Field at an Earth-fixed position [T].

        Parameters
        ----------
        position_ecef_m : numpy.ndarray
            Position in the Earth-fixed frame [m], shape ``(3,)``.

        Returns
        -------
        numpy.ndarray
            Field [T], shape ``(3,)``.
        """
        r = float(np.linalg.norm(position_ecef_m))
        r_hat = position_ecef_m / r
        scale = (IGRF_REFERENCE_RADIUS_M / r) ** 3
        field_nt = scale * (
            3.0 * float(self.moment_nt @ r_hat) * r_hat - self.moment_nt
        )
        return field_nt * 1.0e-9


def load_dipole(
    table_path: str | Path = IGRF_TABLE, epoch_year: float = 2025.0
) -> Dipole:
    """Parse the degree-1 coefficients from the committed IAGA table.

    Parameters
    ----------
    table_path : str or pathlib.Path, optional
        Path to ``igrf14coeffs.txt`` in its upstream format.
    epoch_year : float, optional
        Which epoch column to take, as the decimal year in the table's header
        row. Defaults to the most recent IGRF epoch, 2025.0.

    Returns
    -------
    Dipole
        The tilted dipole at that epoch.

    Raises
    ------
    ValueError
        If the header, the epoch column, or any of the three degree-1
        coefficients is absent — a silently defaulted coefficient would give a
        plausible field with no provenance.
    """
    lines = [
        line
        for line in Path(table_path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = next((line for line in lines if line.split()[0] == "g/h"), None)
    if header is None:
        raise ValueError(f"{table_path}: no 'g/h n m ...' header row")
    columns = header.split()
    label = f"{epoch_year:.1f}"
    if label not in columns:
        raise ValueError(f"{table_path}: no epoch column {label}")
    index = columns.index(label)

    wanted = {("g", "1", "0"): None, ("g", "1", "1"): None, ("h", "1", "1"): None}
    for line in lines:
        parts = line.split()
        key = (parts[0], parts[1], parts[2]) if len(parts) > index else None
        if key in wanted:
            wanted[key] = float(parts[index])
    missing = [k for k, v in wanted.items() if v is None]
    if missing:
        raise ValueError(f"{table_path}: missing degree-1 coefficients {missing}")

    return Dipole(
        moment_nt=np.array(
            [wanted[("g", "1", "1")], wanted[("h", "1", "1")], wanted[("g", "1", "0")]],
            dtype=float,
        ),
        epoch_year=epoch_year,
    )


def circular_orbit_eci_m(
    sma_m: float,
    inc_rad: float,
    raan_rad: float,
    mean_motion_rad_s: float,
    times_s: np.ndarray,
) -> np.ndarray:
    """Inertial positions along a circular orbit.

    Parameters
    ----------
    sma_m : float
        Orbit radius [m].
    inc_rad : float
        Inclination [rad].
    raan_rad : float
        Right ascension of the ascending node [rad].
    mean_motion_rad_s : float
        Mean motion [rad/s].
    times_s : numpy.ndarray
        Times from epoch [s], shape ``(n,)``; argument of latitude is zero at
        ``t = 0``.

    Returns
    -------
    numpy.ndarray
        Inertial positions [m], shape ``(n, 3)``.
    """
    u = mean_motion_rad_s * np.asarray(times_s, dtype=float)
    in_plane = sma_m * np.column_stack((np.cos(u), np.sin(u), np.zeros_like(u)))
    ci, si = np.cos(inc_rad), np.sin(inc_rad)
    co, so = np.cos(raan_rad), np.sin(raan_rad)
    rot = np.array(
        [[co, -so * ci, so * si], [so, co * ci, -co * si], [0.0, si, ci]], dtype=float
    )
    return in_plane @ rot.T


def field_eci_t(
    dipole: Dipole, positions_eci_m: np.ndarray, times_s: np.ndarray
) -> np.ndarray:
    """Field along an inertial trajectory [T].

    The Earth-fixed dipole is rotated into the inertial frame at each time by
    the Earth-rotation angle (zero at ``t = 0``), which is what makes the field
    direction sweep over both the orbital and the diurnal period.

    Parameters
    ----------
    dipole : Dipole
        The Earth-fixed dipole.
    positions_eci_m : numpy.ndarray
        Inertial positions [m], shape ``(n, 3)``.
    times_s : numpy.ndarray
        Matching times from epoch [s], shape ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Inertial fields [T], shape ``(n, 3)``.
    """
    positions = np.asarray(positions_eci_m, dtype=float)
    theta = EARTH_ROTATION_RAD_S * np.asarray(times_s, dtype=float)
    out = np.empty_like(positions)
    for i, (r_eci, ang) in enumerate(zip(positions, theta, strict=True)):
        c, s = np.cos(ang), np.sin(ang)
        eci_from_ecef = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        r_ecef = eci_from_ecef.T @ r_eci
        out[i] = eci_from_ecef @ dipole.field_ecef_t(r_ecef)
    return out
