"""Observability of the attitude-error / gyro-bias model under vector measurements.

The model
---------
The linearised MEKF error state is
:math:`x = [\\delta\\boldsymbol\\theta;\\ \\mathbf b]` — the small attitude
error and the gyro bias, six states — with the standard multiplicative error
dynamics [markley2014, lefferts1982]

.. math::

   \\dot{\\delta\\boldsymbol\\theta} = -[\\boldsymbol\\omega\\times]
       \\delta\\boldsymbol\\theta - \\mathbf b, \\qquad \\dot{\\mathbf b} = 0,

evaluated at zero body rate, so :math:`A = \\begin{bmatrix}0 & -I\\\\ 0 &
0\\end{bmatrix}`. A measurement of a known reference direction
:math:`\\hat r` (unit vector, body frame) has sensitivity
:math:`H = [\\,[\\hat r\\times]\\ \\ 0\\,]`, weighted by the inverse of that
source's angular 1σ.

What it says
------------
* **Two non-parallel vectors** (sun + magnetic field) make the model
  observable, gyro bias included: each skew matrix is rank 2, two non-parallel
  ones stack to rank 3 on the attitude, and the bias enters through
  :math:`HA`, giving rank 6.
* **Near-parallel vectors** collapse the attitude information about the shared
  direction. The metric that says so is
  :math:`\\lambda_{\\min}/\\lambda_{\\max}` of the information matrix
  :math:`M = w_s(I-\\hat s\\hat s^\\top) + w_m(I-\\hat m\\hat m^\\top)` — which
  is **exactly** the quantity the flight Davenport seed gates on
  (``SeedMinObservability``), and at equal weights it is
  :math:`\\sin^2(\\theta/2)`, matching the coarse chain's
  ``MinSinAngle`` = sin 10°. So this module's geometry sweep and the flight
  gate are the same function, which is the point: an analysis that agrees with
  the vehicle by construction cannot drift from it.
* **Eclipse** leaves the magnetometer and the gyro. One vector is rank 2, so
  the model drops to rank 4 of 6 — rotation about :math:`\\hat B` and the bias
  component along it are unobservable. The vehicle does not fall over: the
  filter coasts on the gyro through the eclipse (``MekfMaxCoastSec`` = 300 s is
  *not* what limits it, because the magnetic pair keeps updating), and the
  unobservable direction is re-illuminated at sunrise. The result to take is
  that eclipse is a **rank-deficient** interval by construction, not a fault to
  be detected.

Assumptions
-----------
Zero body rate (a rotating vehicle is *more* observable — the rotation sweeps
the unobservable direction — so this is the conservative case); unit-vector
measurements with isotropic transverse noise; no measurement-time correlation.
The gyro angle random walk does not enter a deterministic observability test:
it is process noise and sets how fast information is *lost* between updates,
not whether it exists.

Units and frames
----------------
Body frame; directions are unit vectors; sigmas [rad]; bias [rad/s]; horizons
[s]. SI throughout.

References
----------
Lefferts, Markley & Shuster, "Kalman Filtering for Spacecraft Attitude
Estimation," *JGCD* 5(5), 1982 [lefferts1982] — the multiplicative error state
and its measurement sensitivity.
Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and
Control*, §6.2 [markley2014].
Design doc §8.1 (attitude determination), §8.2 (sensor fusion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from analysis.control.controllability import RANK_TOLERANCE, numerical_rank, skew
from analysis.control.vehicle import Vehicle

#: Default observability horizon [s]. Ten seconds is a hundred cycles of the
#: 10 Hz GNC rate — long enough for the bias to become observable through the
#: attitude drift it causes, short enough to be a statement about a single
#: acquisition rather than about an orbit.
DEFAULT_HORIZON_S = 10.0


@dataclass(frozen=True)
class ObservabilityResult:
    """Observability of the six-state attitude/bias model in one geometry.

    Attributes
    ----------
    label : str
        What was analysed.
    n_states : int
        State dimension (6).
    rank : int
        Rank of the observability matrix.
    observable : bool
        ``rank == n_states``.
    gramian_min_eig : float
        Smallest eigenvalue of the scaled finite-horizon observability Gramian.
    gramian_max_eig : float
        Largest eigenvalue.
    gramian_condition : float
        ``max/min``; ``inf`` when singular.
    horizon_s : float
        Gramian horizon [s].
    information_ratio : float
        :math:`\\lambda_{\\min}/\\lambda_{\\max}` of the static attitude
        information matrix [-] — the flight ``SeedMinObservability`` metric.
    separation_rad : float
        Angle between the two reference directions [rad]; ``nan`` when only one
        is present.
    """

    label: str
    n_states: int
    rank: int
    observable: bool
    gramian_min_eig: float
    gramian_max_eig: float
    gramian_condition: float
    horizon_s: float
    information_ratio: float
    separation_rad: float


def _error_dynamics() -> np.ndarray:
    """State matrix of the attitude-error/gyro-bias model, shape ``(6, 6)``."""
    a = np.zeros((6, 6))
    a[0:3, 3:6] = -np.eye(3)
    return a


def _measurement_matrix(
    directions: list[np.ndarray], sigmas: list[float]
) -> np.ndarray:
    """Stacked, noise-weighted measurement sensitivity, shape ``(3k, 6)``."""
    rows = []
    for direction, sigma in zip(directions, sigmas, strict=True):
        unit = np.asarray(direction, dtype=float)
        unit = unit / np.linalg.norm(unit)
        rows.append(np.hstack((skew(unit) / sigma, np.zeros((3, 3)))))
    return np.vstack(rows)


def _observability_matrix(a: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Stacked :math:`[H; HA; \\dots; HA^{n-1}]`."""
    blocks = [h]
    for _ in range(a.shape[0] - 1):
        blocks.append(blocks[-1] @ a)
    return np.vstack(blocks)


def _observability_gramian(
    a: np.ndarray, h: np.ndarray, horizon_s: float, steps: int = 400
) -> np.ndarray:
    """Finite-horizon observability Gramian, by quadrature."""
    tau = np.linspace(0.0, horizon_s, steps + 1)
    integrand = np.empty((tau.size, a.shape[0], a.shape[0]))
    for i, t in enumerate(tau):
        h_phi = h @ expm(a * t)
        integrand[i] = h_phi.T @ h_phi
    return np.trapezoid(integrand, tau, axis=0)


def information_ratio(directions: list[np.ndarray], sigmas: list[float]) -> float:
    """Flight-equivalent attitude-observability metric for a vector set.

    :math:`M = \\sum_i w_i (I - \\hat r_i\\hat r_i^\\top)` with
    :math:`w_i = 1/\\sigma_i^2`; the returned value is
    :math:`\\lambda_{\\min}(M)/\\lambda_{\\max}(M)`, the scale-free geometry
    number the Davenport seed gate (``SeedMinObservability``) is written on.

    Parameters
    ----------
    directions : list of numpy.ndarray
        Reference directions, body frame; normalised internally.
    sigmas : list of float
        Matching angular 1σ [rad].

    Returns
    -------
    float
        In [0, 1]. Zero for a single direction (or parallel directions), where
        rotation about it is unobservable.
    """
    m = np.zeros((3, 3))
    for direction, sigma in zip(directions, sigmas, strict=True):
        unit = np.asarray(direction, dtype=float)
        unit = unit / np.linalg.norm(unit)
        m += (np.eye(3) - np.outer(unit, unit)) / sigma**2
    eigenvalues = np.linalg.eigvalsh(m)
    if eigenvalues[-1] <= 0.0:
        return 0.0
    ratio = float(eigenvalues[0] / eigenvalues[-1])
    return ratio if ratio > RANK_TOLERANCE else 0.0


def _analyse(
    label: str,
    directions: list[np.ndarray],
    sigmas: list[float],
    horizon_s: float,
    separation_rad: float,
) -> ObservabilityResult:
    """Common body of the observability entry points."""
    a = _error_dynamics()
    h = _measurement_matrix(directions, sigmas)
    rank = numerical_rank(_observability_matrix(a, h))
    gramian = _observability_gramian(a, h, horizon_s)
    # Scale the bias block by the horizon so the eigenvalues of the two blocks
    # are comparable: a condition number across rad and rad/s would otherwise
    # be a statement about the choice of units.
    scale = np.diag([1.0, 1.0, 1.0, horizon_s, horizon_s, horizon_s])
    eigenvalues = np.clip(np.linalg.eigvalsh(scale @ gramian @ scale.T), 0.0, None)
    lo, hi = float(eigenvalues[0]), float(eigenvalues[-1])
    return ObservabilityResult(
        label=label,
        n_states=6,
        rank=rank,
        observable=rank == 6,
        gramian_min_eig=lo,
        gramian_max_eig=hi,
        gramian_condition=float(hi / lo) if lo > 0.0 else float("inf"),
        horizon_s=horizon_s,
        information_ratio=information_ratio(directions, sigmas),
        separation_rad=separation_rad,
    )


def two_vector_observability(
    vehicle: Vehicle,
    separation_rad: float | None = None,
    *,
    eclipse: bool = False,
    horizon_s: float = DEFAULT_HORIZON_S,
) -> ObservabilityResult:
    """Observability with the sun and magnetic pairs at a given separation.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model; supplies both sigmas and the TRIAD geometry gate.
    separation_rad : float, optional
        Angle between the sun and field directions [rad]. Defaults to 90°, the
        best case. Ignored when ``eclipse`` is set.
    eclipse : bool, optional
        Drop the sun measurement, leaving the magnetometer and the gyro — the
        umbra case.
    horizon_s : float, optional
        Gramian horizon [s].

    Returns
    -------
    ObservabilityResult
        Rank, Gramian conditioning and the flight-equivalent information ratio.
    """
    sun = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 2.0 if separation_rad is None else float(separation_rad)
    field = np.array([np.cos(angle), np.sin(angle), 0.0])

    if eclipse:
        return _analyse(
            "eclipse: magnetometer + gyro",
            [field],
            [vehicle.sensors.sigma_mag_rad],
            horizon_s,
            float("nan"),
        )
    return _analyse(
        f"sun + magnetometer, {np.degrees(angle):.1f} deg apart",
        [sun, field],
        [vehicle.sensors.sigma_sun_rad, vehicle.sensors.sigma_mag_rad],
        horizon_s,
        angle,
    )


def vector_geometry_sweep(vehicle: Vehicle, separations_rad: np.ndarray) -> np.ndarray:
    """Information ratio across a sweep of sun/field separations.

    The curve this returns is what the ``SeedMinObservability`` and
    ``MinSinAngle`` gates cut: it falls to zero as the two directions become
    parallel, and the flight software refuses a solve below its threshold
    rather than reporting a fix whose roll about the shared direction is
    unconstrained.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model; supplies the two sigmas.
    separations_rad : numpy.ndarray
        Separation angles [rad], shape ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Information ratios [-], shape ``(n,)``.
    """
    sun = np.array([1.0, 0.0, 0.0])
    sigmas = [vehicle.sensors.sigma_sun_rad, vehicle.sensors.sigma_mag_rad]
    return np.array(
        [
            information_ratio([sun, np.array([np.cos(a), np.sin(a), 0.0])], sigmas)
            for a in np.asarray(separations_rad, dtype=float)
        ]
    )
