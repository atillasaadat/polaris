"""Actuator envelope geometry — the spine of the sizing tool (design doc §7, §12).

Everything :mod:`analysis.sizing` reports rests on one identity,
:math:`h(\\mathbf u) = a_{\\max}\\lVert W^\\top\\mathbf u\\rVert_1`, and on the
claim that the inscribed radius is found exactly by enumerating facet normals.
Both are checked here against closed forms and against an independent method
before the module is pointed at a vehicle.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.sizing.envelope import (
    DegenerateArrayError,
    envelope,
    fibonacci_sphere,
    sampled_inscribed,
    support,
)

#: Three orthogonal wheels — the case whose every radius is exact by hand.
ORTHOGONAL = np.eye(3)

#: The reference vehicle's layout: four wheels on the body diagonals.
PYRAMID = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
    ]
).T / np.sqrt(3.0)


def test_orthogonal_triad_has_the_closed_form_radii():
    """A cube: ``r_in`` is exactly ``h_max``, ``r_out`` exactly ``h_max*sqrt(3)``.

    :math:`h(\\mathbf u) = h_{\\max}(|u_x|+|u_y|+|u_z|)`, whose minimum over the
    unit sphere is 1 (at a body axis) and whose maximum is :math:`\\sqrt3` (at a
    body diagonal). If the facet enumeration is wrong, this is where it shows.
    """
    result = envelope(ORTHOGONAL, 0.5)
    assert result.inscribed == pytest.approx(0.5)
    assert result.circumscribed == pytest.approx(0.5 * np.sqrt(3.0), rel=1e-3)
    assert result.per_body_axis == pytest.approx(np.full(3, 0.5))
    assert result.ellipsoid_semi_axes == pytest.approx(np.full(3, 0.5))


def test_the_four_wheel_pyramid_matches_its_hand_calculation():
    """Per-body-axis reach ``4*h_max/sqrt(3)``; inscribed radius ``4*h_max/sqrt(6)``.

    Both by hand. Each spin axis has :math:`|w_i\\cdot\\mathbf e_x| = 1/\\sqrt3`
    and there are four, so a body axis reaches :math:`4h_{\\max}/\\sqrt3`. The
    weakest direction is the facet normal
    :math:`(\\mathbf w_1\\times\\mathbf w_2)` normalised, i.e.
    :math:`(0,-1,1)/\\sqrt2`, in which two wheels contribute nothing and two
    contribute :math:`2/\\sqrt6` each — :math:`4h_{\\max}/\\sqrt6`, 29 % less than
    the body-axis figure. Sizing on the body-axis number would claim capability
    the array does not have.
    """
    h_max = 0.5
    result = envelope(PYRAMID, h_max)
    assert result.per_body_axis == pytest.approx(
        np.full(3, 4.0 * h_max / np.sqrt(3.0)), rel=1e-4
    )
    assert result.inscribed == pytest.approx(4.0 * h_max / np.sqrt(6.0), rel=1e-4)
    assert result.worst_direction @ np.array([1.0, 0.0, 0.0]) == pytest.approx(
        0.0, abs=1e-9
    )
    # Isotropic in the L2 sense: W W^T = (4/3) I, so every semi-axis is equal.
    assert result.ellipsoid_semi_axes == pytest.approx(
        np.full(3, h_max * np.sqrt(4.0 / 3.0)), rel=1e-4
    )


@pytest.mark.parametrize(
    "axes",
    [
        ORTHOGONAL,
        PYRAMID,
        # A deliberately skewed five-wheel set: no symmetry to rescue either method.
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.6, 0.7, 0.4],
                [-0.2, 0.3, 0.93],
            ]
        ).T,
    ],
)
def test_facet_enumeration_and_dense_sampling_agree(axes):
    """The exact inscribed radius, cross-checked by a method sharing no reasoning.

    Sampling converges to the answer from **above** — a finite sample rarely
    lands exactly on the worst direction — so the exact value must be at or below
    the sampled one, and the gap bounds the sampling error. A facet enumeration
    that missed the true minimum would come out *above* the sample and fail the
    first assertion, which is the assertion that matters.

    The tolerance on the second is 2 %, and deliberately loose: the support
    function has a **kink** at a facet normal (it is a sum of absolute values),
    so the sampled minimum errs *linearly* in the angular sample spacing rather
    than quadratically as it would at a smooth minimum. At 200000 points the
    spacing is ~0.008 rad and the residual is a few tenths of a percent — while
    the errors this test exists to catch are far larger: taking a body-axis
    reach instead of the true minimum is 29 % high on the pyramid, and taking
    ``r_out`` is 41 % high.
    """
    exact = envelope(axes, 1.0)
    sampled, _ = sampled_inscribed(axes, 1.0, count=200000)
    assert exact.inscribed <= sampled + 1e-12
    assert exact.inscribed == pytest.approx(sampled, rel=2e-2)


@pytest.mark.parametrize("axes", [ORTHOGONAL, PYRAMID])
def test_the_ellipsoid_is_inscribed_in_the_zonotope(axes):
    """:math:`\\lVert W^\\top u\\rVert_2 \\le \\lVert W^\\top u\\rVert_1` in every
    direction.

    The L2 ellipsoid is the image of the L2 ball, the zonotope is the image of
    the L∞ box, and the ball is inside the box — so the ellipsoid can never stick
    out. Checked as the support-function inequality in 20000 directions rather
    than asserted, because it is the property that makes "which allocator do you
    fly" a sizing question.
    """
    directions = fibonacci_sphere(20000)
    zonotope = support(axes, 1.0, directions)
    ellipsoid = np.linalg.norm(directions @ axes, axis=1)
    assert np.all(ellipsoid <= zonotope + 1e-12)


def test_the_support_identity_matches_brute_force():
    """``h(u) = a_max * ||W^T u||_1`` against an explicit maximisation over corners.

    The identity is the module's whole computation, so it is checked once against
    the definition it claims to be: the largest :math:`\\mathbf u^\\top W\\mathbf a`
    over every corner of the command box.
    """
    rng = np.random.default_rng(20260808)
    axes = rng.normal(size=(3, 4))
    axes /= np.linalg.norm(axes, axis=0)
    corners = np.array(np.meshgrid(*[[-1.0, 1.0]] * 4)).reshape(4, -1).T
    for direction in rng.normal(size=(25, 3)):
        unit = direction / np.linalg.norm(direction)
        brute = float(np.max((corners * 0.75) @ (axes.T @ unit)))
        assert support(axes, 0.75, unit) == pytest.approx(brute)


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        (np.eye(3)[:, :2], "at least 3"),
        (
            # Three axes all in the x-y plane: no z authority at all.
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]).T,
            "coplanar",
        ),
        (
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).T,
            "coplanar",
        ),
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).T,
            "zero vector",
        ),
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, np.nan]]).T, "finite"),
    ],
)
def test_degenerate_layouts_are_refused(axes, match):
    """A layout that cannot span three axes raises rather than returning a number.

    Returning zero would read as "a very small margin" on a report; the truth is
    "this vehicle cannot be controlled about one axis", which is not a margin at
    all.
    """
    with pytest.raises(DegenerateArrayError, match=match):
        envelope(axes, 1.0)


def test_a_non_positive_capacity_is_refused():
    """A zero or negative per-actuator limit is a config defect, not an envelope."""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="capacity must be positive"):
            envelope(ORTHOGONAL, bad)


def test_radii_scale_linearly_with_capacity():
    """Every radius is homogeneous of degree one in the per-actuator limit."""
    small = envelope(PYRAMID, 1.0)
    large = envelope(PYRAMID, 7.0)
    assert large.inscribed == pytest.approx(7.0 * small.inscribed)
    assert large.circumscribed == pytest.approx(7.0 * small.circumscribed)
    assert large.ellipsoid_semi_axes == pytest.approx(7.0 * small.ellipsoid_semi_axes)


def test_the_fibonacci_sphere_is_unit_and_deterministic():
    """Directions are unit vectors and do not depend on an RNG."""
    first = fibonacci_sphere(500)
    assert np.linalg.norm(first, axis=1) == pytest.approx(np.ones(500))
    assert fibonacci_sphere(500) == pytest.approx(first)
