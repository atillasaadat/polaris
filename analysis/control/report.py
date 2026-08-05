"""Verdicts for the control analysis: REQ-ACTL-006, -007 and -008.

Separated from the modules that *measure* so the measurement and the judgement
are not the same file. Two levels are offered, and they are for different
readers:

* :func:`margin_report` returns :class:`MarginReport`, the domain-shaped result
  — per-axis margins with their own accessors. This is what margin-specific
  tests assert on.
* :func:`control_analysis_report` returns the shared
  :class:`analysis.common.report.AnalysisReport`, one flat criterion list across
  all three requirements, with provenance and assumptions attached. This is the
  standing convention every analysis tool follows, the thing rendered to text
  beside the figures, and what the requirement-verifying tests assert on.

References
----------
Design doc §8.5, §13; ``docs/requirements/adcs_control.rst``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.common.report import AnalysisReport, Criterion
from analysis.control.controllability import (
    orbit_averaged_mtq_controllability,
    wheel_controllability,
    wheel_failure_subsets,
)
from analysis.control.margins import (
    MAX_SENSITIVITY_PEAK,
    MIN_GAIN_MARGIN_DB,
    MIN_PHASE_MARGIN_DEG,
    AxisMargins,
    axis_margins,
)
from analysis.control.observability import two_vector_observability
from analysis.control.plant import MAX_SISO_COUPLING_RATIO
from analysis.control.vehicle import Vehicle

#: Smallest acceptable rank, as a fraction of the state dimension. Rank is an
#: integer and the criterion is really "full rank", but expressing it as a ratio
#: lets it share the numeric criterion machinery rather than needing a second
#: kind of check for one case.
FULL_RANK = 1.0


@dataclass(frozen=True)
class MarginReport:
    """Per-axis margin result for a vehicle, with the pass/fail verdict.

    Attributes
    ----------
    vehicle_name : str
        The vehicle the margins were measured on.
    sampled : bool
        Whether the sampled-data loop was analysed.
    axes : tuple of AxisMargins
        One entry per body axis, in x, y, z order.
    thresholds : dict
        The requirement values applied, so a report carries its own criteria.
    """

    vehicle_name: str
    sampled: bool
    axes: tuple[AxisMargins, ...]
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "min_gain_margin_db": MIN_GAIN_MARGIN_DB,
            "min_phase_margin_deg": MIN_PHASE_MARGIN_DEG,
            "max_sensitivity_peak": MAX_SENSITIVITY_PEAK,
        }
    )

    @property
    def passes(self) -> bool:
        """Every axis meets every threshold."""
        return all(a.passes for a in self.axes)

    @property
    def siso_assumption_holds(self) -> bool:
        """Every axis's per-axis model is trustworthy for this configuration."""
        return all(a.siso_assumption_holds for a in self.axes)

    def failures(self) -> list[str]:
        """Human-readable reasons, one per violated threshold.

        Returns
        -------
        list of str
            Empty when :attr:`passes`.
        """
        out: list[str] = []
        for a in self.axes:
            if not a.stable:
                out.append(f"axis {a.axis}: closed loop is unstable")
                continue
            if a.gain_margin_db < MIN_GAIN_MARGIN_DB:
                out.append(
                    f"axis {a.axis}: gain margin {a.gain_margin_db:.2f} dB "
                    f"< {MIN_GAIN_MARGIN_DB} dB "
                    f"(up {a.gain_margin_up_db:.2f}, down {a.gain_margin_down_db:.2f})"
                )
            if a.phase_margin_deg < MIN_PHASE_MARGIN_DEG:
                out.append(
                    f"axis {a.axis}: phase margin {a.phase_margin_deg:.2f} deg "
                    f"< {MIN_PHASE_MARGIN_DEG} deg"
                )
            if a.sensitivity_peak > MAX_SENSITIVITY_PEAK:
                out.append(
                    f"axis {a.axis}: sensitivity peak {a.sensitivity_peak:.2f} "
                    f"> {MAX_SENSITIVITY_PEAK}"
                )
        return out


def margin_report(
    vehicle: Vehicle,
    *,
    sampled: bool = True,
    integrator: bool = True,
    computation_delay_cycles: int = 0,
) -> MarginReport:
    """Margins on all three body axes, with the requirement verdict.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    sampled : bool, optional
        Analyse the sampled-data loop (default).
    integrator : bool, optional
        ``False`` analyses the saturated-loop case.
    computation_delay_cycles : int, optional
        Extra whole cycles of transport delay.

    Returns
    -------
    MarginReport
        Per-axis margins and the pass/fail verdict against
        :data:`analysis.control.margins.MIN_GAIN_MARGIN_DB`,
        :data:`analysis.control.margins.MIN_PHASE_MARGIN_DEG` and
        :data:`analysis.control.margins.MAX_SENSITIVITY_PEAK`.
    """
    return MarginReport(
        vehicle_name=vehicle.name,
        sampled=sampled,
        axes=tuple(
            axis_margins(
                vehicle,
                i,
                sampled=sampled,
                integrator=integrator,
                computation_delay_cycles=computation_delay_cycles,
            )
            for i in range(3)
        ),
    )


#: The modelling assumptions every number in the control analysis rests on.
#: Carried into the report because a margin without its assumptions is not a
#: result (design doc §8.5).
ASSUMPTIONS = (
    "Small angle: the error rotation is radians-small; the short-way-round "
    "branch of the flight law is inactive.",
    "Unsaturated: neither the body-torque limit nor the per-wheel torque box is "
    "reached, so both direction-preserving scales are the identity.",
    "Integrator active: the conditional integrator is not frozen. The frozen "
    "(saturated) loop is the Ki = 0 case and is better behaved at low frequency.",
    "Wheels only: the MTQ/MAG duty-cycle interlock gates magnetometer samples, "
    "not the wheel torque path, and the pointing loop closes on wheels.",
    "Rigid body, ideal actuators and sensors: no wheel-motor lag, no gyro "
    "dynamics, no flexible modes.",
    "Per-axis (SISO): a diagonal inertia tensor (enforced at load) and small "
    "stored wheel momentum (measured; see warnings). Design doc SS8.5, "
    "'SISO validity boundary and MIMO roadmap'.",
)


def _margin_criteria(report: MarginReport) -> list[Criterion]:
    """One gain, phase and sensitivity criterion per body axis."""
    out: list[Criterion] = []
    for a in report.axes:
        out.append(
            Criterion(
                name=f"gain margin (axis {a.axis})",
                requirement="REQ-ACTL-006",
                threshold=MIN_GAIN_MARGIN_DB,
                measured=a.gain_margin_db,
                units="dB",
                sense="min",
                note=(
                    f"conditionally stable: tolerates {a.gain_margin_up_db:.1f} dB up, "
                    f"{a.gain_margin_down_db:.1f} dB down"
                    if a.conditionally_stable
                    else ""
                ),
            )
        )
        out.append(
            Criterion(
                name=f"phase margin (axis {a.axis})",
                requirement="REQ-ACTL-006",
                threshold=MIN_PHASE_MARGIN_DEG,
                measured=a.phase_margin_deg,
                units="deg",
                sense="min",
                note=f"crossover {a.gain_crossover_rad_s:.4f} rad/s",
            )
        )
        out.append(
            Criterion(
                name=f"sensitivity peak (axis {a.axis})",
                requirement="REQ-ACTL-006",
                threshold=MAX_SENSITIVITY_PEAK,
                measured=a.sensitivity_peak,
                units="-",
                sense="max",
                note=(
                    f"disk margin {a.disk_margin:.2f}: "
                    f"{a.disk_gain_margin_db:.1f} dB and "
                    f"{a.disk_phase_margin_deg:.1f} deg simultaneously"
                ),
            )
        )
    return out


def _controllability_criteria(vehicle: Vehicle) -> list[Criterion]:
    """Full-rank and conditioning criteria for the wheel and rod configurations."""
    out: list[Criterion] = []
    full = wheel_controllability(vehicle)
    out.append(
        Criterion(
            name="wheel array rank (all wheels)",
            requirement="REQ-ACTL-007",
            threshold=FULL_RANK,
            measured=full.kalman_rank / full.n_states,
            units="of full",
            sense="min",
            note=f"conditioning of A A^T = {full.input_conditioning:.3f} (isotropic = 1)",
        )
    )
    for wheels, result in wheel_failure_subsets(vehicle).items():
        label = ",".join(str(w) for w in wheels)
        out.append(
            Criterion(
                name=f"wheels {label} rank (one failed)",
                requirement="REQ-ACTL-007",
                threshold=FULL_RANK,
                measured=result.kalman_rank / result.n_states,
                units="of full",
                sense="min",
            )
        )
        out.append(
            Criterion(
                name=f"wheels {label} conditioning",
                requirement="REQ-ACTL-007",
                threshold=vehicle.alloc_min_conditioning,
                measured=result.input_conditioning,
                units="-",
                sense="min",
                note="threshold is the flight allocator's own AllocMinConditioning gate",
            )
        )
    orbit = orbit_averaged_mtq_controllability(vehicle)
    out.append(
        Criterion(
            name="magnetorquers, orbit-averaged rank",
            requirement="REQ-ACTL-007",
            threshold=FULL_RANK,
            measured=orbit.kalman_rank / orbit.n_states,
            units="of full",
            sense="min",
            note="instantaneously rank 4 of 6 by construction; m x B has no "
            "component along B-hat",
        )
    )
    return out


def _observability_criteria(vehicle: Vehicle) -> list[Criterion]:
    """Full-rank criteria for the nominal geometry and the documented eclipse case."""
    nominal = two_vector_observability(vehicle)
    gate = two_vector_observability(
        vehicle, float(np.arcsin(vehicle.sensors.min_sin_angle))
    )
    eclipse = two_vector_observability(vehicle, eclipse=True)
    return [
        Criterion(
            name="attitude + gyro bias rank (sun 90 deg from field)",
            requirement="REQ-ACTL-008",
            threshold=FULL_RANK,
            measured=nominal.rank / nominal.n_states,
            units="of full",
            sense="min",
            note=f"information ratio {nominal.information_ratio:.4f}",
        ),
        Criterion(
            name="attitude + gyro bias rank (at the MinSinAngle gate)",
            requirement="REQ-ACTL-008",
            threshold=FULL_RANK,
            measured=gate.rank / gate.n_states,
            units="of full",
            sense="min",
            note=(
                f"information ratio {gate.information_ratio:.2e} at "
                f"{np.degrees(gate.separation_rad):.1f} deg — observable, but this is "
                "where the flight TRIAD gate refuses"
            ),
        ),
        Criterion(
            name="eclipse rank (magnetometer + gyro)",
            requirement="REQ-ACTL-008",
            threshold=eclipse.n_states - 2,
            measured=eclipse.rank,
            units="states",
            sense="min",
            note="rank 4 of 6 is the expected and documented result: rotation about "
            "the field line and the bias along it are unobservable by construction",
        ),
    ]


def _warnings(vehicle: Vehicle, report: MarginReport) -> tuple[str, ...]:
    """Conditions that qualify the analysis without failing it."""
    out: list[str] = []
    for a in report.axes:
        if a.siso_assumption_holds or math.isnan(a.siso_coupling_ratio):
            continue
        out.append(
            f"axis {a.axis}: with the wheel array at its momentum capacity the "
            f"gyroscopic cross-coupling at the {a.gain_crossover_rad_s:.3f} rad/s "
            f"crossover is {a.siso_coupling_ratio:.1f}x the diagonal term, against a "
            f"{MAX_SISO_COUPLING_RATIO:g} limit for per-axis analysis. These margins "
            f"hold for stored momentum below {a.siso_momentum_limit_nms:.2e} N.m.s "
            f"({100.0 * a.siso_momentum_limit_nms / vehicle.wheel_max_momentum_nms:.2g}% "
            "of one wheel's capacity); momentum-biased operation needs the MIMO "
            "analysis committed in design doc SS8.5."
        )
    return tuple(out)


def control_analysis_report(
    vehicle: Vehicle,
    config_path: str | Path,
    *,
    sampled: bool = True,
    computation_delay_cycles: int = 0,
) -> AnalysisReport:
    """The full control analysis as one structured, renderable report.

    Covers REQ-ACTL-006 (margins), -007 (controllability) and -008
    (observability), with the configuration provenance and the modelling
    assumptions attached.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    config_path : str or pathlib.Path
        The config it was loaded from; recorded as provenance.
    sampled : bool, optional
        Analyse the sampled-data loop (default).
    computation_delay_cycles : int, optional
        Extra whole cycles of transport delay.

    Returns
    -------
    analysis.common.report.AnalysisReport
        Criteria for all three requirements, plus warnings.
    """
    margins = margin_report(
        vehicle, sampled=sampled, computation_delay_cycles=computation_delay_cycles
    )
    gains = vehicle.pid
    return AnalysisReport(
        title=f"Linear control analysis — {vehicle.name}",
        config_path=str(config_path),
        provenance={
            "loop": (
                f"sampled-data at {vehicle.control_period_s:g} s"
                if sampled
                else "continuous-time idealisation"
            ),
            "delay": f"{computation_delay_cycles} cycle(s) of transport delay",
            "gains": (
                f"Kp={gains.kp_nm_per_rad:g} N.m/rad, "
                f"Ki={gains.ki_nm_per_rad_s:g} N.m/(rad.s), "
                f"Kd={gains.kd_nm_per_radps:g} N.m/(rad/s)"
            ),
            "inertia": (
                "diag("
                + ", ".join(f"{j:g}" for j in vehicle.principal_moments_kgm2)
                + ") kg.m^2"
            ),
            "actuators": (
                f"{vehicle.wheel_spin_axes.shape[1]} wheels at "
                f"{vehicle.wheel_max_torque_nm:g} N.m / "
                f"{vehicle.wheel_max_momentum_nms:g} N.m.s, "
                f"{vehicle.mtq_axes.shape[1]} rods at "
                f"{vehicle.mtq_max_dipole_am2:g} A.m^2"
            ),
        },
        assumptions=ASSUMPTIONS,
        criteria=tuple(
            _margin_criteria(margins)
            + _controllability_criteria(vehicle)
            + _observability_criteria(vehicle)
        ),
        warnings=_warnings(vehicle, margins),
    )
