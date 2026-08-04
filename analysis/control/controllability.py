"""Controllability of the as-flown actuator suite.

Three questions, three answers
------------------------------
1. **The four-wheel pyramid** spans three axes with a one-dimensional null
   space; the six-state attitude model is controllable and well conditioned.
2. **Every 3-of-4 wheel-failure subset** is still controllable — that is the
   whole point of the body-diagonal geometry — but not equally well: losing a
   wheel costs authority anisotropically, and the Gramian says by how much.
3. **The magnetorquers alone** are instantaneously **rank 2**: the torque
   :math:`\\vec m\\times\\vec B` can never have a component along
   :math:`\\hat B`, so at any instant one axis is uncontrollable. What redeems
   them is that :math:`\\hat B` turns over an orbit, making the *time-varying*
   Gramian full rank. This is the standard result for magnetic attitude control
   and the reason B-dot is an asymptotic law rather than a fast one
   [avanzini2012].

Metrics, and why these
----------------------
* **Kalman rank** of :math:`[B\\;AB\\;\\dots\\;A^{n-1}B]` — the binary answer.
  Reported with a relative singular-value tolerance, because a rank test on a
  floating-point matrix without one is a coin flip near the boundary.
* **Finite-horizon controllability Gramian**
  :math:`W(T)=\\int_0^T e^{A\\tau}BB^\\top e^{A^\\top\\tau}\\,d\\tau`. The
  infinite-horizon Gramian does not exist here: the double integrator is not
  Hurwitz, so ``control.gram`` cannot be used and a horizon must be named.
* **Input-map conditioning** :math:`\\lambda_{\\min}/\\lambda_{\\max}` of
  :math:`A_wA_w^\\top` — the *same scale-free geometry metric* the flight
  allocator gates on (``RwAllocationConfig::min_conditioning``,
  ``AllocMinConditioning`` = 0.05 in the reference config), so a subset this
  module calls badly conditioned is one the flight allocator would refuse.

Non-dimensionalisation, stated because it changes the numbers
-------------------------------------------------------------
The state mixes radians and rad/s and the input mixes N·m across differently
rated actuators, so a raw Gramian's condition number is a statement about the
choice of units. Both are scaled before the eigenvalues are taken: the rate
block by the horizon (so it reads in "radians per horizon"), and each input
column by that actuator's rating (so an input of magnitude 1 is one wheel at
its torque box, or one rod at its rated moment). The **rank** conclusions are
invariant to this; the conditioning numbers are not, and are only comparable
between cases scaled the same way.

Units and frames
----------------
Body frame throughout; torques [N·m], dipoles [A·m²], fields [T], angles [rad],
rates [rad/s], horizons [s].

References
----------
Kalman, as presented in Åström & Murray, *Feedback Systems*, §6.2 and §7.2
[astrom2008] — the rank test and the Gramian.
Wie, *Space Vehicle Dynamics and Control*, 2nd ed., §7.4 [wie2008] — wheel
array geometry and redundancy.
Avanzini & Giulietti [avanzini2012] — the instantaneous rank deficiency of
magnetic actuation and the orbit-timescale convergence it forces.
Design doc §7 (actuators), §8.5 (control).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import expm

from analysis.control.field import (
    circular_orbit_eci_m,
    field_eci_t,
    load_dipole,
)
from analysis.control.vehicle import Vehicle

#: Relative singular-value tolerance for every rank test here. A singular value
#: below this fraction of the largest is treated as zero.
RANK_TOLERANCE = 1.0e-9

#: Default Gramian horizon for the wheel studies [s]. Chosen as roughly the
#: closed-loop settling time the committed gains give (≈28 s, §8.5) times three,
#: i.e. the span over which the loop actually manoeuvres the vehicle.
DEFAULT_HORIZON_S = 100.0


@dataclass(frozen=True)
class ControllabilityResult:
    """Controllability of one actuator configuration.

    Attributes
    ----------
    label : str
        What was analysed, e.g. ``"wheels 0,1,2"`` or ``"MTQ, instantaneous"``.
    n_states : int
        State dimension (6 for the attitude model).
    kalman_rank : int
        Rank of the controllability matrix.
    controllable : bool
        ``kalman_rank == n_states``.
    gramian_min_eig : float
        Smallest eigenvalue of the scaled finite-horizon Gramian [-].
    gramian_max_eig : float
        Largest eigenvalue [-].
    gramian_condition : float
        ``max/min``; ``inf`` when the Gramian is singular.
    horizon_s : float
        The Gramian horizon [s].
    input_min_singular : float
        Smallest singular value of the (unscaled) body-torque input map
        [N·m per unit command], zero for a rank-deficient actuator set.
    input_conditioning : float
        :math:`\\lambda_{\\min}/\\lambda_{\\max}` of :math:`A A^\\top` for the
        actuator axes [-] — directly comparable with the flight allocator's
        ``AllocMinConditioning`` gate.
    """

    label: str
    n_states: int
    kalman_rank: int
    controllable: bool
    gramian_min_eig: float
    gramian_max_eig: float
    gramian_condition: float
    horizon_s: float
    input_min_singular: float
    input_conditioning: float


def _attitude_a() -> np.ndarray:
    """State matrix of the linearised attitude model, shape ``(6, 6)``."""
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    return a


def numerical_rank(matrix: np.ndarray) -> int:
    """Numerical rank at :data:`RANK_TOLERANCE`, relative to the largest value."""
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular > RANK_TOLERANCE * singular[0]))


def _kalman_rank(a: np.ndarray, b: np.ndarray) -> int:
    """Rank of :math:`[B\\;AB\\;\\dots\\;A^{n-1}B]`."""
    n = a.shape[0]
    blocks = [b]
    for _ in range(n - 1):
        blocks.append(a @ blocks[-1])
    return numerical_rank(np.hstack(blocks))


def _state_scaling(horizon_s: float) -> np.ndarray:
    """Diagonal state scaling: rate measured in radians per horizon."""
    return np.diag([1.0, 1.0, 1.0, horizon_s, horizon_s, horizon_s])


def _gramian_lti(
    a: np.ndarray, b: np.ndarray, horizon_s: float, steps: int = 400
) -> np.ndarray:
    """Finite-horizon controllability Gramian of an LTI pair, by quadrature."""
    tau = np.linspace(0.0, horizon_s, steps + 1)
    integrand = np.empty((tau.size, a.shape[0], a.shape[0]))
    for i, t in enumerate(tau):
        phi_b = expm(a * t) @ b
        integrand[i] = phi_b @ phi_b.T
    return np.trapezoid(integrand, tau, axis=0)


def _gramian_ltv(a: np.ndarray, b_of_t: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Finite-horizon Gramian of a time-varying input map.

    :math:`W = \\int \\Phi(t_f,\\tau)B(\\tau)B(\\tau)^\\top\\Phi(t_f,\\tau)^\\top
    d\\tau` with :math:`\\Phi` the (constant-``A``) transition matrix.
    """
    t_f = float(times_s[-1])
    integrand = np.empty((times_s.size, a.shape[0], a.shape[0]))
    for i, t in enumerate(times_s):
        phi_b = expm(a * (t_f - t)) @ b_of_t[i]
        integrand[i] = phi_b @ phi_b.T
    return np.trapezoid(integrand, times_s, axis=0)


def _summarise(
    label: str,
    a: np.ndarray,
    b: np.ndarray,
    gramian: np.ndarray,
    horizon_s: float,
    axes: np.ndarray,
) -> ControllabilityResult:
    """Assemble a :class:`ControllabilityResult` from the computed pieces."""
    scale = _state_scaling(horizon_s)
    eigenvalues = np.linalg.eigvalsh(scale @ gramian @ scale.T)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    lo, hi = float(eigenvalues[0]), float(eigenvalues[-1])
    gram = np.asarray(axes @ axes.T, dtype=float)
    gram_eigs = np.linalg.eigvalsh(gram)
    conditioning = float(gram_eigs[0] / gram_eigs[-1]) if gram_eigs[-1] > 0.0 else 0.0
    torque_singular = np.linalg.svd(axes, compute_uv=False)
    rank = _kalman_rank(a, b)
    return ControllabilityResult(
        label=label,
        n_states=a.shape[0],
        kalman_rank=rank,
        controllable=rank == a.shape[0],
        gramian_min_eig=lo,
        gramian_max_eig=hi,
        gramian_condition=float(hi / lo) if lo > 0.0 else float("inf"),
        horizon_s=horizon_s,
        input_min_singular=float(torque_singular[-1]) if axes.shape[1] >= 3 else 0.0,
        input_conditioning=max(conditioning, 0.0),
    )


def wheel_controllability(
    vehicle: Vehicle,
    wheels: tuple[int, ...] | None = None,
    horizon_s: float = DEFAULT_HORIZON_S,
) -> ControllabilityResult:
    """Controllability of the attitude model driven by a wheel subset.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    wheels : tuple of int, optional
        Wheel indices retained. ``None`` uses every installed wheel.
    horizon_s : float, optional
        Gramian horizon [s].

    Returns
    -------
    ControllabilityResult
        Rank, Gramian conditioning and the array geometry metric.
    """
    axes = vehicle.wheel_torque_axes(wheels)
    a = _attitude_a()
    # Input scaled by the wheel rating: a unit input is one wheel at its box.
    b = np.vstack(
        (
            np.zeros((3, axes.shape[1])),
            vehicle.inertia_inverse() @ axes * vehicle.wheel_max_torque_nm,
        )
    )
    label = "wheels " + ",".join(
        str(i) for i in (wheels if wheels is not None else range(axes.shape[1]))
    )
    return _summarise(label, a, b, _gramian_lti(a, b, horizon_s), horizon_s, axes)


def wheel_failure_subsets(
    vehicle: Vehicle, horizon_s: float = DEFAULT_HORIZON_S, keep: int = 3
) -> dict[tuple[int, ...], ControllabilityResult]:
    """Controllability of every ``keep``-of-N wheel subset.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    horizon_s : float, optional
        Gramian horizon [s].
    keep : int, optional
        Number of surviving wheels. ``3`` is the single-failure case the
        pyramid geometry exists for.

    Returns
    -------
    dict
        Keyed by the surviving wheel indices, in ascending order.
    """
    count = vehicle.wheel_spin_axes.shape[1]
    return {
        subset: wheel_controllability(vehicle, subset, horizon_s)
        for subset in combinations(range(count), keep)
    }


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix :math:`[v]_\\times`, shape ``(3, 3)``."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def _mtq_input_map(vehicle: Vehicle, field_body_t: np.ndarray) -> np.ndarray:
    """Body-torque map from a commanded dipole in a given field, shape ``(3, 3)``.

    :math:`\\vec\\tau = \\vec m\\times\\vec B = -[\\vec B]_\\times\\vec m`, so
    the map is singular along :math:`\\hat B` by construction — no tuning, no
    configuration and no fault can change that.
    """
    return -skew(np.asarray(field_body_t, dtype=float))


def mtq_controllability(
    vehicle: Vehicle,
    field_body_t: np.ndarray,
    horizon_s: float = DEFAULT_HORIZON_S,
) -> ControllabilityResult:
    """Controllability from the magnetorquers in one **frozen** field.

    The expected result is rank 4 of 6: the torque map is rank 2, so the
    reachable set never includes rotation about :math:`\\hat B`. This is the
    baseline the orbit-averaged case is measured against, not a fault.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    field_body_t : numpy.ndarray
        Body-frame magnetic field [T], shape ``(3,)``.
    horizon_s : float, optional
        Gramian horizon [s].

    Returns
    -------
    ControllabilityResult
        Rank and conditioning; ``gramian_condition`` is ``inf`` by design.
    """
    axes = _mtq_input_map(vehicle, field_body_t) * vehicle.mtq_max_dipole_am2
    a = _attitude_a()
    b = np.vstack((np.zeros((3, 3)), vehicle.inertia_inverse() @ axes))
    return _summarise(
        "MTQ, frozen field", a, b, _gramian_lti(a, b, horizon_s), horizon_s, axes
    )


def orbit_averaged_mtq_controllability(
    vehicle: Vehicle, samples: int = 721, orbits: float = 1.0
) -> ControllabilityResult:
    """Controllability from the magnetorquers over a whole orbit.

    The field direction turns as the vehicle moves and as the tilted dipole
    rotates under it, so the *time-varying* Gramian is full rank even though
    every instantaneous input map is rank 2. The reported conditioning is the
    honest price: the weakest direction is orders of magnitude weaker than the
    strongest, which is why magnetic-only control is slow and why B-dot is
    written as an asymptotic law.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model; supplies the orbit and the rod rating.
    samples : int, optional
        Quadrature points over the span. Odd, so the trapezoid rule brackets
        the mid-orbit point.
    orbits : float, optional
        Span in orbital periods.

    Returns
    -------
    ControllabilityResult
        ``kalman_rank`` is reported from the *stacked* instantaneous maps —
        the time-varying analogue of the LTI rank test — and the Gramian
        eigenvalues come from the time-varying integral.
    """
    orbit = vehicle.orbit
    horizon_s = orbits * orbit.period_s
    times = np.linspace(0.0, horizon_s, samples)
    positions = circular_orbit_eci_m(
        orbit.sma_m, orbit.inc_rad, orbit.raan_rad, orbit.mean_motion_rad_s, times
    )
    fields = field_eci_t(load_dipole(), positions, times)

    a = _attitude_a()
    j_inv = vehicle.inertia_inverse()
    maps = np.empty((samples, 6, 3))
    stacked = np.empty((3, 3 * samples))
    for i, field in enumerate(fields):
        axes = _mtq_input_map(vehicle, field) * vehicle.mtq_max_dipole_am2
        maps[i] = np.vstack((np.zeros((3, 3)), j_inv @ axes))
        stacked[:, 3 * i : 3 * i + 3] = axes

    gramian = _gramian_ltv(a, maps, times)
    result = _summarise(
        f"MTQ, {orbits:g} orbit(s) averaged",
        a,
        maps[0],
        gramian,
        horizon_s,
        stacked,
    )
    # The LTI rank test on one frozen map answers the wrong question here; the
    # time-varying rank is the rank of the Gramian, which is what full
    # controllability over the span means. Taken on the *scaled* Gramian, since
    # an unscaled one spans T^3 to T and its numerical rank would be a statement
    # about the horizon rather than about the system.
    scale = _state_scaling(horizon_s)
    rank = numerical_rank(scale @ gramian @ scale.T)
    return ControllabilityResult(
        label=result.label,
        n_states=result.n_states,
        kalman_rank=rank,
        controllable=rank == result.n_states,
        gramian_min_eig=result.gramian_min_eig,
        gramian_max_eig=result.gramian_max_eig,
        gramian_condition=result.gramian_condition,
        horizon_s=horizon_s,
        input_min_singular=result.input_min_singular,
        input_conditioning=result.input_conditioning,
    )
