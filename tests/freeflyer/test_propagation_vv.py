"""FreeFlyer vs GMAT-golden propagation cross-validation (REQ-VV-006).

Each golden case from ``tests/golden/gmat_propagation.json`` is replayed in
FreeFlyer (``tools/freeflyer/vv.py``) and the sampled states compared against
the fixture. The C++ stack verifies against the same fixture in
``polaris_golden_tests``, which makes the comparison three-way: a failure here
with the C++ side green isolates FreeFlyer-vs-GMAT; a simultaneous failure
points at the fixture or a shared convention.

Tolerances are **FreeFlyer-specific** — wider than the fixture's own
GMAT-vs-Polaris bands because a third implementation brings its own gravity
coefficients, DE ephemeris, EOP handling, and the ICRF-vs-MJ2000Eq frame bias
(≲1 m at LEO radius). Each band is the worst disagreement measured on
FreeFlyer 7.10.1 with a factor 4-10 of margin, and the measured values are
recorded beside them so drift is visible in review, not just in CI red.
"""

from __future__ import annotations

import math

import pytest

#: name -> (position tol [m], velocity tol [m/s], measured worst dr [m] on 7.10.1)
FF_TOLERANCES = {
    # Mu pinned to wgs84::kGM on both sides: integrator truncation + frame
    # bias only.
    "two_body": (0.05, 5.0e-5, 0.005),
    # J2 from FreeFlyer's own coefficient set vs GMAT's EGM96 J2 (identical to
    # published precision).
    "zonal_j2": (0.10, 2.0e-4, 0.015),
    # DE440 (Polaris/GMAT) vs FreeFlyer's DE file for Sun/Moon positions.
    "third_body": (0.75, 8.0e-4, 0.15),
    # Degree-8 field: EGM2008 vs FreeFlyer's default geopotential
    # coefficients.
    "iss_leo": (1.0, 1.5e-3, 0.22),
    "sso_leo": (0.75, 8.0e-4, 0.15),
    # A GEO day of SRP: FreeFlyer's solar flux is a fixed 1358 W/m^2 against
    # GMAT's scripted 1361, and the penumbra models differ; both scale with
    # season, hence the widest band.
    "geo": (2.0, 1.0e-4, 0.27),
    "molniya_heo": (0.60, 5.0e-4, 0.12),
    # Kinematic constant-rate rotation is exact in both tools; the band covers
    # the carrier orbit only.
    "attitude_spinner": (0.05, 5.0e-5, 0.002),
}

#: Attitude agreement band [deg]; measured 0.0 (below double precision).
ATTITUDE_TOL_DEG = 1.0e-3


@pytest.mark.freeflyer
@pytest.mark.verifies("REQ-VV-006")
@pytest.mark.parametrize("case_name", sorted(FF_TOLERANCES))
def test_freeflyer_reproduces_golden_case(ff_case_results, case_name):
    case, results = ff_case_results(case_name)
    pos_tol, vel_tol, _measured = FF_TOLERANCES[case_name]

    assert len(results) == len(case["samples"])
    worst_dr = worst_dv = 0.0
    for got, want in zip(results, case["samples"]):
        # The harness lands on the golden sample times themselves; a drifted
        # time would masquerade as a huge state error, so pin it first.
        assert abs(got["t_s"] - want["t_s"]) < 1.0e-5, "sample-time mismatch"
        worst_dr = max(worst_dr, math.dist(got["position_m"], want["position_m"]))
        worst_dv = max(worst_dv, math.dist(got["velocity_m_s"], want["velocity_m_s"]))

    assert worst_dr < pos_tol, (
        f"{case_name}: FreeFlyer diverges from the GMAT golden by {worst_dr:.3f} m "
        f"(band {pos_tol} m). polaris_golden_tests green = FreeFlyer-vs-GMAT issue; "
        f"red = fixture or shared-convention issue."
    )
    assert (
        worst_dv < vel_tol
    ), f"{case_name}: velocity diverges {worst_dv:.2e} m/s (band {vel_tol})"


@pytest.mark.freeflyer
@pytest.mark.verifies("REQ-VV-006")
def test_freeflyer_reproduces_golden_attitude(ff_case_results):
    case, results = ff_case_results("attitude_spinner")
    worst_deg = 0.0
    for got, want in zip(results, case["samples"]):
        dot = abs(
            sum(
                a * b
                for a, b in zip(got["attitude_quaternion"], want["attitude_quaternion"])
            )
        )
        worst_deg = max(worst_deg, 2.0 * math.degrees(math.acos(min(1.0, dot))))
    assert worst_deg < ATTITUDE_TOL_DEG, (
        f"attitude diverges by {worst_deg:.2e} deg — check the vector-first/"
        f"scalar-last FreeFlyer quaternion convention before anything else"
    )
