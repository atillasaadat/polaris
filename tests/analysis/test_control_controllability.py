"""Controllability of the as-flown actuator suite (REQ-ACTL-007).

Covers the three cases that decide whether the vehicle can be pointed: the full
pyramid, every single-wheel-failure subset, and the magnetorquers — whose
instantaneous rank deficiency is a property of :math:`\\vec m\\times\\vec B`
and is asserted as such, not treated as a fault.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from analysis.control import (
    mtq_controllability,
    orbit_averaged_mtq_controllability,
    wheel_controllability,
    wheel_failure_subsets,
)
from analysis.control.field import (
    circular_orbit_eci_m,
    field_eci_t,
    load_dipole,
)

#: Sample field used for the frozen-field case [T], body frame: a mid-latitude
#: LEO field of ~34 uT with all three components populated, so a rank result
#: cannot come from an axis-aligned special case.
FROZEN_FIELD_T = np.array([2.0e-5, 1.0e-5, 2.5e-5])


@pytest.mark.verifies("REQ-ACTL-007")
def test_four_wheel_pyramid_is_controllable_and_isotropic(vehicle, record_property):
    """The body-diagonal pyramid spans three axes with an isotropic torque map."""
    result = wheel_controllability(vehicle)
    assert result.controllable
    assert result.kalman_rank == 6
    assert result.gramian_min_eig > 0.0
    assert np.isfinite(result.gramian_condition)
    # lambda_min/lambda_max of A A^T is exactly 1 for the body diagonals — the
    # claim the config makes in its own comment, checked rather than believed.
    assert result.input_conditioning == pytest.approx(1.0, abs=1e-9)
    record_property(
        "margin_pct",
        f"{100.0 * result.input_conditioning / vehicle.alloc_min_conditioning:.0f}",
    )


@pytest.mark.verifies("REQ-ACTL-007")
def test_every_three_of_four_wheel_subset_is_controllable(vehicle):
    """Losing any one wheel leaves three-axis control, above the flight gate."""
    subsets = wheel_failure_subsets(vehicle)
    assert len(subsets) == 4
    for wheels, result in subsets.items():
        assert result.controllable, wheels
        assert result.kalman_rank == 6
        assert result.gramian_min_eig > 0.0
        # The surviving trio is a regular simplex leg set: 1/4 of the isotropic
        # conditioning, still five times the flight allocator's own gate, so a
        # subset this suite calls controllable is one the vehicle would fly.
        assert result.input_conditioning == pytest.approx(0.25, rel=1e-6)
        assert result.input_conditioning > vehicle.alloc_min_conditioning
    # Redundancy costs authority: the smallest reachable direction shrinks.
    full = wheel_controllability(vehicle)
    for result in subsets.values():
        assert result.gramian_min_eig < full.gramian_min_eig


def test_a_collinear_wheel_array_is_uncontrollable(vehicle):
    """A degenerate array must be flagged, not quietly pseudo-inverted."""
    collinear = replace(
        vehicle,
        wheel_spin_axes=np.column_stack(
            [np.array([0.0, 0.0, 1.0])] * vehicle.wheel_spin_axes.shape[1]
        ),
    )
    result = wheel_controllability(collinear)
    assert not result.controllable
    assert result.kalman_rank == 2  # one torque axis, its angle and its rate
    assert result.gramian_min_eig == pytest.approx(0.0, abs=1e-12)
    assert result.input_conditioning == pytest.approx(0.0, abs=1e-12)
    assert result.input_conditioning < vehicle.alloc_min_conditioning


@pytest.mark.verifies("REQ-ACTL-007")
def test_magnetorquers_are_instantaneously_rank_deficient(vehicle):
    """m x B has no component along B, so one axis is uncontrollable at any instant."""
    result = mtq_controllability(vehicle, FROZEN_FIELD_T)
    assert not result.controllable
    assert result.kalman_rank == 4  # 6 states less the angle and rate about B-hat
    assert result.gramian_min_eig == pytest.approx(0.0, abs=1e-9)
    assert not np.isfinite(result.gramian_condition)


@pytest.mark.verifies("REQ-ACTL-007")
def test_magnetorquers_are_controllable_averaged_over_an_orbit(vehicle):
    """The field direction turning over an orbit restores full rank."""
    result = orbit_averaged_mtq_controllability(vehicle)
    assert result.controllable
    assert result.kalman_rank == 6
    assert result.gramian_min_eig > 0.0
    assert result.horizon_s == pytest.approx(vehicle.orbit.period_s)
    # Full rank, but far from isotropic — which is why magnetic-only control is
    # an orbit-timescale process and B-dot is written as an asymptotic law.
    assert result.gramian_condition > 10.0
    wheels = wheel_controllability(vehicle)
    assert result.gramian_condition > wheels.gramian_condition


def test_the_dipole_model_reproduces_the_field_band_the_config_states(vehicle):
    """Degree-1 field along the reference orbit sits inside the documented band.

    ``leo_smallsat.yaml`` states the IGRF magnitude runs ~22–52 uT over this
    500 km SSO. The degree-1 truncation used for the controllability study
    should land inside that band and span most of it; a coefficient parsed from
    the wrong column, or a sign error in the dipole, would not.
    """
    orbit = vehicle.orbit
    times = np.linspace(0.0, orbit.period_s, 721)
    positions = circular_orbit_eci_m(
        orbit.sma_m, orbit.inc_rad, orbit.raan_rad, orbit.mean_motion_rad_s, times
    )
    magnitudes = np.linalg.norm(field_eci_t(load_dipole(), positions, times), axis=1)
    assert 20.0e-6 < magnitudes.min() < 26.0e-6
    assert 44.0e-6 < magnitudes.max() < 55.0e-6
    # The polar/equatorial factor of two is the dipole's signature.
    assert magnitudes.max() / magnitudes.min() == pytest.approx(2.0, rel=0.05)


def test_the_dipole_coefficients_come_from_the_committed_table():
    """The three degree-1 Gauss coefficients are read, not restated."""
    dipole = load_dipole()
    assert dipole.epoch_year == 2025.0
    # g10 dominates and is negative: the geomagnetic axis points roughly south.
    assert dipole.moment_nt[2] < -25000.0
    assert abs(dipole.moment_nt[2]) > 5.0 * np.linalg.norm(dipole.moment_nt[:2])
    # The dipole tilt from the rotation axis is the ~9.5 deg every textbook quotes.
    tilt_deg = np.degrees(
        np.arctan2(np.linalg.norm(dipole.moment_nt[:2]), abs(dipole.moment_nt[2]))
    )
    assert 8.0 < tilt_deg < 11.0


def test_a_missing_epoch_column_is_refused(tmp_path):
    """A silently defaulted coefficient would be a plausible field with no source."""
    table = tmp_path / "coeffs.txt"
    table.write_text("g/h n m 2020.0\ng  1  0 -29404.8\n")
    with pytest.raises(ValueError, match="no epoch column"):
        load_dipole(table, epoch_year=2025.0)
