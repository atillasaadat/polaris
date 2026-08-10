"""Actuator envelope geometry: what an array can deliver, and in which direction.

An array of :math:`N` single-axis actuators with axes :math:`\\mathbf w_i` as the
columns of :math:`W \\in \\mathbb R^{3\\times N}`, each limited to
:math:`|a_i| \\le a_{\\max}`, delivers the **zonotope**

.. math::

   Z = \\{ W\\mathbf a : \\mathbf a \\in [-a_{\\max}, a_{\\max}]^N \\},

whose support function in a unit direction :math:`\\mathbf u` is

.. math::

   h(\\mathbf u) = \\max_{\\mathbf x \\in Z} \\mathbf u^\\top \\mathbf x
                 = a_{\\max}\\,\\lVert W^\\top\\mathbf u\\rVert_1 .

That identity is the whole module — every number below is a maximum or a minimum
of it. It holds because each coordinate of :math:`\\mathbf a` is chosen
independently, so the best choice is
:math:`a_i = a_{\\max}\\operatorname{sgn}(\\mathbf w_i\\cdot\\mathbf u)`.

The two radii, and why only one of them sizes anything
------------------------------------------------------
* **Inscribed radius** :math:`r_{\\mathrm{in}} = \\min_{\\mathbf u} h(\\mathbf u)`
  — the capability **guaranteed in every direction**. Sizing is done against
  this, always. It is the radius of the largest ball inside :math:`Z`.
* **Circumscribed radius** :math:`r_{\\mathrm{out}} = \\max_{\\mathbf u}
  h(\\mathbf u)` — the capability in the array's *best* direction, attained at a
  vertex. Reporting :math:`r_{\\mathrm{out}}` as though it were capability is the
  classic sizing error: "the wheels can do this in the best direction" and "the
  wheels can do this in every direction" differ by a factor of
  :math:`\\sqrt2` on the reference body-diagonal pyramid (1.15 against 0.82
  N·m·s), and without bound on a poorly spread array.

The **ellipsoid** :math:`E = \\{W\\mathbf a : \\lVert\\mathbf a\\rVert_2 \\le
a_{\\max}\\}` has semi-axes :math:`a_{\\max}\\sigma_i` from the singular values of
:math:`W`. It is what a minimum-norm (L2 pseudo-inverse) allocator reaches under
an **RMS** command limit, and it is **inscribed in the zonotope** — the L2 ball
is inside the L∞ box, so its image is inside the box's image. The reference
vehicle flies the L∞ allocator (``AllocMethodSel: 1``), which reaches the
zonotope; the ellipsoid is reported because it is the bound that applies if the
allocation method ever changes.

Degenerate layouts are **refused**, not approximated: fewer than three actuators,
coplanar or collinear axes, or any :math:`W` whose smallest singular value is
negligible against its largest cannot span three axes, and a number computed for
one of those would be a confident description of an uncontrollable vehicle.

Units and frames
----------------
Unit-free in the actuator quantity: pass momentum capacity [N·m·s] for a
momentum envelope, torque capacity [N·m] for a torque envelope, dipole [A·m²]
for a rod set. Axes are unit vectors in the **body frame**; every radius is in
the same units as the capacity handed in.

References
----------
Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and
Control*, §7.3 and §7.5 [markley2014] — reaction-wheel arrays, the pyramid
configuration, and the momentum envelope of a redundant array.
Wie, *Space Vehicle Dynamics and Control*, 2nd ed., §7.4 [wie2008] — momentum
management and the wheel-array capability envelope.
Sidi, *Spacecraft Dynamics and Control*, §7.3 [sidi1997] — momentum-storage
sizing against the envelope rather than against a per-unit rating.
Design doc §7 (actuators, W-matrix assembly), §12 (analysis tools).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

#: Smallest :math:`\sigma_{\min}/\sigma_{\max}` of the axis matrix accepted as
#: spanning three axes [-]. Below it the array is coplanar to within
#: double-precision noise and no envelope exists in the collapsed direction.
#: Deliberately far looser than the flight allocator's own
#: ``AllocMinConditioning`` gate: this refuses the *geometrically impossible*,
#: while the flight gate refuses the *badly conditioned*, and conflating the two
#: would make this module reject arrays it is meant to report a poor margin on.
MIN_AXIS_CONDITIONING = 1.0e-9

#: Points on the Fibonacci sphere used by the sampled cross-check. Large enough
#: that the sampled minimum lands within ~1 % of the exact facet answer. The
#: convergence is only first-order in the angular sample spacing, because the
#: support function has a kink at the facet normal where the minimum sits — see
#: the agreement test, which states the resulting tolerance and why it is loose.
SPHERE_SAMPLES = 20000


class DegenerateArrayError(ValueError):
    """The actuator axes do not span three dimensions.

    Raised rather than returning a radius, because every number this module
    produces would otherwise describe a vehicle that cannot be controlled about
    one axis as though it merely had a small margin there.
    """


@dataclass(frozen=True)
class Envelope:
    """The achievable set of an actuator array, as radii and per-axis reach.

    Attributes
    ----------
    axes : numpy.ndarray
        Actuator axes as columns, shape ``(3, N)``, unit vectors in the body
        frame.
    capacity : float
        Per-actuator limit, in the quantity being enveloped (N·m·s, N·m, A·m²).
    inscribed : float
        :math:`r_{\\mathrm{in}}`, the capability guaranteed in **every**
        direction. **This is the number sizing is done against.**
    circumscribed : float
        :math:`r_{\\mathrm{out}}`, the capability in the array's best direction.
        Never a sizing number.
    ellipsoid_semi_axes : numpy.ndarray
        Semi-axes of the L2 (minimum-norm) ellipsoid, shape ``(3,)``, descending.
        The smallest is the guaranteed radius of that ellipsoid, and it is
        always ``<= inscribed``.
    per_body_axis : numpy.ndarray
        :math:`h(\\mathbf e_x), h(\\mathbf e_y), h(\\mathbf e_z)`, shape ``(3,)``
        — the reach along each body axis. Larger than :attr:`inscribed` unless a
        body axis happens to be the worst direction.
    worst_direction : numpy.ndarray
        The unit direction attaining :attr:`inscribed`, shape ``(3,)``. Reported
        because "which way is the array weakest" is what a layout change acts
        on.
    """

    axes: np.ndarray
    capacity: float
    inscribed: float
    circumscribed: float
    ellipsoid_semi_axes: np.ndarray
    per_body_axis: np.ndarray
    worst_direction: np.ndarray

    @property
    def n_actuators(self) -> int:
        """Number of installed actuators."""
        return int(self.axes.shape[1])

    @property
    def ellipsoid_inscribed(self) -> float:
        """Smallest ellipsoid semi-axis — the L2 allocator's guaranteed radius."""
        return float(self.ellipsoid_semi_axes[-1])


def support(axes: np.ndarray, capacity: float, direction: np.ndarray) -> np.ndarray:
    """Zonotope support :math:`h(\\mathbf u)=a_{\\max}\\lVert W^\\top\\mathbf u
    \\rVert_1`.

    Parameters
    ----------
    axes : numpy.ndarray
        Actuator axes as columns, shape ``(3, N)``.
    capacity : float
        Per-actuator limit.
    direction : numpy.ndarray
        One direction ``(3,)`` or many as rows ``(k, 3)``. Normalised here, so
        the caller may pass unnormalised directions.

    Returns
    -------
    numpy.ndarray
        Support value(s); a 0-d array for a single direction.
    """
    u = np.atleast_2d(np.asarray(direction, dtype=float))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    values = capacity * np.sum(np.abs(u @ axes), axis=1)
    return values[0] if np.ndim(direction) == 1 else values


def fibonacci_sphere(count: int = SPHERE_SAMPLES) -> np.ndarray:
    """Near-uniform directions on the unit sphere, shape ``(count, 3)``.

    The golden-angle spiral: deterministic (no RNG to seed), and with no
    clustering at the poles that a naive latitude/longitude grid would put
    exactly where a body-axis-aligned array is strongest.

    Parameters
    ----------
    count : int, optional
        Number of directions.

    Returns
    -------
    numpy.ndarray
        Unit vectors as rows, shape ``(count, 3)``.
    """
    i = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / count
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def _facet_normals(axes: np.ndarray) -> np.ndarray:
    """Candidate directions attaining :math:`\\min_{\\mathbf u} h(\\mathbf u)`.

    A zonotope's facets are spanned by pairs of generators, so every facet
    normal is :math:`\\mathbf w_i \\times \\mathbf w_j` normalised. The minimum
    of the support function over the sphere is attained at a facet normal: the
    inscribed ball touches :math:`\\partial Z` on a facet (it cannot touch only
    an edge or a vertex of a centrally symmetric convex body without also
    touching the facets meeting there), and the support in a facet's normal
    direction is that facet's distance from the centre.

    Exact for any :math:`N`, at :math:`\\binom{N}{2}` candidates — trivial for
    the wheel counts a spacecraft carries, which is why no optimiser appears
    here.

    Parameters
    ----------
    axes : numpy.ndarray
        Actuator axes as columns, shape ``(3, N)``.

    Returns
    -------
    numpy.ndarray
        Unit normals as rows; parallel pairs (which span no facet) are dropped.
    """
    normals = []
    for i, j in combinations(range(axes.shape[1]), 2):
        n = np.cross(axes[:, i], axes[:, j])
        norm = float(np.linalg.norm(n))
        if norm > 1.0e-12:
            normals.append(n / norm)
    if not normals:
        raise DegenerateArrayError(
            "every pair of actuator axes is parallel: the array spans one line"
        )
    return np.asarray(normals, dtype=float)


def check_axes(axes: np.ndarray) -> np.ndarray:
    """Validate an axis matrix and return it as unit columns.

    Parameters
    ----------
    axes : numpy.ndarray
        Actuator axes as columns, shape ``(3, N)``.

    Returns
    -------
    numpy.ndarray
        The same axes with each column normalised, shape ``(3, N)``.

    Raises
    ------
    DegenerateArrayError
        If there are fewer than three actuators, if any axis is the zero vector
        or non-finite, or if the axes are coplanar (rank < 3, judged by
        :data:`MIN_AXIS_CONDITIONING`). Three axes is the algebraic minimum for
        three-axis control; anything less cannot produce an envelope with a
        non-zero inscribed radius, and returning zero would read as a very small
        margin rather than as an impossible one.
    """
    w = np.atleast_2d(np.asarray(axes, dtype=float))
    if w.ndim != 2 or w.shape[0] != 3:
        raise DegenerateArrayError(f"axes must have shape (3, N), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise DegenerateArrayError("axes contain non-finite values")
    if w.shape[1] < 3:
        raise DegenerateArrayError(
            f"{w.shape[1]} actuator(s): three-axis control needs at least 3 axes"
        )
    norms = np.linalg.norm(w, axis=0)
    if np.any(norms <= 0.0):
        raise DegenerateArrayError("an actuator axis is the zero vector")
    unit = w / norms
    singular = np.linalg.svd(unit, compute_uv=False)
    if singular[-1] <= MIN_AXIS_CONDITIONING * singular[0]:
        raise DegenerateArrayError(
            f"actuator axes are coplanar or collinear "
            f"(sigma_min/sigma_max = {singular[-1] / singular[0]:.3e}); the array "
            "spans fewer than three axes and has no three-dimensional envelope"
        )
    return unit


def envelope(axes: np.ndarray, capacity: float) -> Envelope:
    """Build the achievable-set description of an actuator array.

    Parameters
    ----------
    axes : numpy.ndarray
        Actuator axes as columns, shape ``(3, N)``, body frame. Normalised here.
    capacity : float
        Per-actuator limit in the quantity being enveloped — momentum [N·m·s],
        torque [N·m] or dipole [A·m²]. Must be positive and finite.

    Returns
    -------
    Envelope
        Radii, per-body-axis reach and the worst direction, all in the units of
        @p capacity.

    Raises
    ------
    DegenerateArrayError
        On a layout that does not span three axes (see :func:`check_axes`).
    ValueError
        On a non-positive or non-finite capacity.
    """
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError(f"capacity must be positive and finite, got {capacity!r}")
    w = check_axes(axes)

    normals = _facet_normals(w)
    facet_values = support(w, capacity, normals)
    best = int(np.argmin(facet_values))
    inscribed = float(facet_values[best])

    # The maximum is attained at a vertex, i.e. in the direction of one of the
    # 2^(N-1) sign-vector sums. Enumerating those is exponential; the support
    # function is maximised where the generators add most coherently, which the
    # facet-normal set does not necessarily contain — so this one is taken over
    # the dense sample plus the facet normals plus the generator directions
    # themselves, which brackets it tightly and is only ever *reported*.
    candidates = np.vstack((normals, w.T, fibonacci_sphere()))
    all_values = support(w, capacity, candidates)

    return Envelope(
        axes=w,
        capacity=float(capacity),
        inscribed=inscribed,
        circumscribed=float(np.max(all_values)),
        ellipsoid_semi_axes=capacity * np.linalg.svd(w, compute_uv=False),
        per_body_axis=support(w, capacity, np.eye(3)),
        worst_direction=normals[best],
    )


def sampled_inscribed(axes: np.ndarray, capacity: float, count: int = SPHERE_SAMPLES):
    """Inscribed radius by dense spherical sampling — the independent check.

    Same quantity as :attr:`Envelope.inscribed` by a method sharing none of its
    reasoning: sample the sphere and take the smallest support. It converges to
    the exact answer from **above** (a sample rarely lands exactly on the worst
    direction), so ``sampled >= exact`` always, and the gap bounds the sampling
    error. ``tests/analysis/test_sizing_envelope.py`` asserts the two agree.

    Parameters
    ----------
    axes : numpy.ndarray
        Actuator axes as columns, shape ``(3, N)``.
    capacity : float
        Per-actuator limit.
    count : int, optional
        Sample count.

    Returns
    -------
    tuple of (float, numpy.ndarray)
        The sampled minimum and the direction attaining it.
    """
    w = check_axes(axes)
    directions = fibonacci_sphere(count)
    values = support(w, capacity, directions)
    i = int(np.argmin(values))
    return float(values[i]), directions[i]
