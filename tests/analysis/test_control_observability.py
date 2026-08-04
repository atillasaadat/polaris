"""Observability of the attitude-error / gyro-bias model (REQ-ACTL-008).

The load-bearing check is that this analysis and the flight software compute
the *same* geometry metric: the information ratio here must reproduce the
closed form the reference config's ``SeedMinObservability`` derivation is
written on, so the two cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.control import (
    information_ratio,
    two_vector_observability,
    vector_geometry_sweep,
)


@pytest.mark.verifies("REQ-ACTL-008")
def test_two_non_parallel_vectors_make_attitude_and_bias_observable(
    vehicle, record_property
):
    """Sun + magnetometer + gyro observes all six states, bias included."""
    result = two_vector_observability(vehicle)
    assert result.observable
    assert result.rank == 6
    assert result.gramian_min_eig > 0.0
    assert np.isfinite(result.gramian_condition)
    assert result.information_ratio > 0.0
    gate = vehicle.sensors.min_sin_angle
    at_gate = two_vector_observability(vehicle, float(np.arcsin(gate)))
    record_property(
        "margin_pct",
        f"{100.0 * result.information_ratio / at_gate.information_ratio:.0f}",
    )


@pytest.mark.verifies("REQ-ACTL-008")
def test_information_ratio_matches_the_flight_closed_form(vehicle):
    """The analysis metric is the flight metric, checked against its own algebra.

    ``leo_smallsat.yaml`` derives ``SeedMinObservability`` from

        lambda_min/lambda_max = [1 - sqrt(1 - 4 w_s w_m sin^2(th)/(w_s+w_m)^2)]/2

    with ``w = 1/sigma^2``. Reproducing that here for the vehicle's own sigmas
    is what keeps the two from drifting apart.
    """
    w_s = 1.0 / vehicle.sensors.sigma_sun_rad**2
    w_m = 1.0 / vehicle.sensors.sigma_mag_rad**2
    sun = np.array([1.0, 0.0, 0.0])
    for angle_deg in (5.0, 10.0, 30.0, 60.0, 90.0):
        angle = np.deg2rad(angle_deg)
        field = np.array([np.cos(angle), np.sin(angle), 0.0])
        expected = (
            1.0 - np.sqrt(1.0 - 4.0 * w_s * w_m * np.sin(angle) ** 2 / (w_s + w_m) ** 2)
        ) / 2.0
        measured = information_ratio(
            [sun, field],
            [vehicle.sensors.sigma_sun_rad, vehicle.sensors.sigma_mag_rad],
        )
        assert measured == pytest.approx(expected, rel=1e-9)


def test_equal_weights_collapse_to_the_half_angle_form(vehicle):
    """At equal sigmas the metric is sin^2(theta/2) — the 0.0076-at-10-deg line."""
    unit = np.array([1.0, 0.0, 0.0])
    for angle_deg in (10.0, 45.0, 90.0):
        angle = np.deg2rad(angle_deg)
        other = np.array([np.cos(angle), np.sin(angle), 0.0])
        measured = information_ratio([unit, other], [0.01, 0.01])
        assert measured == pytest.approx(np.sin(angle / 2.0) ** 2, rel=1e-9)
    assert information_ratio(
        [unit, np.array([np.cos(np.deg2rad(10.0)), np.sin(np.deg2rad(10.0)), 0.0])],
        [0.01, 0.01],
    ) == pytest.approx(0.0076, abs=5e-5)


@pytest.mark.verifies("REQ-ACTL-008")
def test_near_parallel_vectors_degrade_toward_unobservable(vehicle):
    """Roll about the shared direction is what is lost, monotonically."""
    separations = np.deg2rad(np.array([1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0]))
    ratios = vector_geometry_sweep(vehicle, separations)
    assert np.all(np.diff(ratios) > 0.0)
    assert ratios[0] < ratios[-1] / 100.0
    # Exactly parallel: no information about rotation about the shared axis.
    sun = np.array([1.0, 0.0, 0.0])
    sigmas = [vehicle.sensors.sigma_sun_rad, vehicle.sensors.sigma_mag_rad]
    assert information_ratio([sun, sun.copy()], sigmas) == 0.0
    # The flight TRIAD gate cuts the curve where the ratio is still small but
    # non-zero: the refusal is a design choice about usable accuracy, not a
    # singularity the algebra forced.
    gate_ratio = float(
        vector_geometry_sweep(
            vehicle, np.array([np.arcsin(vehicle.sensors.min_sin_angle)])
        )[0]
    )
    assert 0.0 < gate_ratio < 0.01


@pytest.mark.verifies("REQ-ACTL-008")
def test_eclipse_leaves_the_model_rank_deficient(vehicle):
    """One vector plus a gyro is rank 4 of 6 — by construction, not by fault."""
    result = two_vector_observability(vehicle, eclipse=True)
    assert not result.observable
    assert result.rank == 4
    assert result.gramian_min_eig == pytest.approx(0.0, abs=1e-9)
    assert result.information_ratio == 0.0
    assert np.isnan(result.separation_rad)
