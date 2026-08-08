"""The sizing verdict: every criterion, its provenance, and what was assumed.

Two levels, as in :mod:`analysis.control.report`:

* :func:`sizing_analysis` returns :class:`SizingAnalysis`, the domain-shaped
  result — the envelopes, the disturbance budget, the drivers and the derived
  parameters — which is what the sizing-specific tests and the plots read.
* :func:`sizing_report` returns the shared
  :class:`analysis.common.report.AnalysisReport`, one flat criterion list with
  provenance and assumptions attached. That is the standing convention, the thing
  rendered beside the figures, and what the CLI gates on.

The measuring modules each own their criteria (``wheels.criteria``,
``magnetorquers.criteria``, ``parameters.criteria``) because each judgement
belongs beside the physics that produced it; this module assembles them, attaches
the provenance and the assumptions, and adds the warnings — including the
standing one that **no requirement in the baseline is written on actuator
sizing**, so none of these criteria carries an ID it could be mistaken for
verification evidence of.

References
----------
Design doc §7, §8.5, §12; ``analysis/CLAUDE.md`` (the reporting convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.common.report import AnalysisReport, Criterion
from analysis.control.vehicle import Vehicle
from analysis.sizing import magnetorquers, parameters, wheels
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.disturbances import DisturbanceBudget, disturbance_budget
from analysis.sizing.magnetorquers import MtqSizing
from analysis.sizing.parameters import DerivedParameter
from analysis.sizing.wheels import WheelSizing

#: A commanded wheel-torque limit below this fraction of the installed wheel's
#: catalog capability is warned about as a deep derate [-]. Half is the point at
#: which "derated for a reason" and "wrong unit" stop being distinguishable
#: without an argument on record; it is a prompt, never a verdict.
DERATE_WARN_FRACTION = 0.5


@dataclass(frozen=True)
class SizingAnalysis:
    """Everything the sizing analysis computed, before it is judged.

    Attributes
    ----------
    vehicle : Vehicle
        The as-flown model.
    assumptions : SizingAssumptions
        The assumptions in force.
    budget : DisturbanceBudget
        The §5.3 disturbance-torque budget.
    wheels : WheelSizing
        Wheel envelopes and momentum drivers.
    mtq : MtqSizing
        Rod authority and the B-dot noise floor.
    derived : tuple of DerivedParameter
        The flight tuning this design implies, each with its justification.
    """

    vehicle: Vehicle
    assumptions: SizingAssumptions
    budget: DisturbanceBudget
    wheels: WheelSizing
    mtq: MtqSizing
    derived: tuple[DerivedParameter, ...]

    def criteria(self) -> tuple[Criterion, ...]:
        """Every pass/fail criterion, in report order."""
        return tuple(
            wheels.criteria(self.wheels, self.assumptions)
            + magnetorquers.criteria(self.vehicle, self.mtq, self.assumptions)
            + parameters.criteria(self.vehicle, self.wheels, self.mtq, self.assumptions)
        )


def sizing_analysis(
    vehicle: Vehicle, assumptions: SizingAssumptions | None = None
) -> SizingAnalysis:
    """Run the whole sizing computation for one vehicle.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model, from
        :func:`analysis.control.vehicle.load_vehicle`.
    assumptions : SizingAssumptions, optional
        Defaults to :class:`~analysis.sizing.assumptions.SizingAssumptions`.

    Returns
    -------
    SizingAnalysis

    Raises
    ------
    analysis.sizing.envelope.DegenerateArrayError
        If either actuator array fails to span three dimensions — refused rather
        than reported with a misleading radius.
    """
    assumptions = assumptions or SizingAssumptions()
    budget = disturbance_budget(vehicle, assumptions)
    # The rods are sized first, because whether they can remove the tip-off
    # decides whether the wheels have to. A vehicle with demonstrated magnetic
    # detumble authority (M2) flies rods-then-wheels, so its wheels are judged
    # on the D1b handover; one without has no such option and D1 binds.
    mtq_result = magnetorquers.mtq_sizing(vehicle, budget, assumptions)
    magnetic_detumble = (
        mtq_result.removable_momentum_nms
        >= assumptions.margin
        * float(np.max(vehicle.principal_moments_kgm2))
        * assumptions.tipoff_rate_radps
    )
    wheel_result = wheels.wheel_sizing(
        vehicle, budget, assumptions, magnetic_detumble=magnetic_detumble
    )
    return SizingAnalysis(
        vehicle=vehicle,
        assumptions=assumptions,
        budget=budget,
        wheels=wheel_result,
        mtq=mtq_result,
        derived=parameters.derived_parameters(
            vehicle, wheel_result, mtq_result, assumptions
        ),
    )


def _warnings(analysis: SizingAnalysis) -> tuple[str, ...]:
    """Conditions that qualify the analysis without failing it."""
    vehicle = analysis.vehicle
    out = [
        "No requirement in the baseline is written on actuator sizing, so no "
        "criterion here carries a requirement ID. REQ-ACTL-009 governs the "
        "stored-momentum envelope as a behaviour and REQ-ACTL-010 the "
        "desaturation law, neither of which is a bound on how large the "
        "actuators must be. Writing one — 'the wheel array shall provide, in "
        "every direction, at least 1.3x the momentum required by the sizing "
        "drivers' — is the gap this tool exposes.",
        "The aerodynamic term is assumption-dominated: thermospheric density at "
        f"{analysis.budget.altitude_m / 1000.0:.0f} km varies by more than an "
        "order of magnitude over the solar cycle, far more than any other "
        "uncertainty in this budget. The static exponential model used here is "
        "the plant's own baseline, not its NRLMSIS truth model.",
    ]

    if analysis.wheels.envelope_limited:
        out.append(
            f"The binding momentum limit is the flight envelope, not the wheels: "
            f"MomentumEnvelopeNms = {vehicle.momentum_envelope_nms:g} N.m.s "
            f"against a hardware inscribed radius of "
            f"{analysis.wheels.momentum.inscribed:.3g} N.m.s "
            f"({analysis.wheels.momentum.inscribed / vehicle.momentum_envelope_nms:.0f}x "
            "larger). Every momentum criterion is judged on the smaller number, "
            "because momentum the vehicle raises an envelope event over is "
            "momentum it does not have."
        )

    commanded = analysis.wheels.commanded_torque_nm
    catalog = analysis.wheels.catalog_torque_nm
    if commanded < DERATE_WARN_FRACTION * catalog:
        out.append(
            f"The commanded wheel-torque limit is a deep derate: WheelMaxTorqueNm "
            f"= {commanded:.3g} N.m against a catalog capability of "
            f"{catalog:.3g} N.m ({catalog / commanded:.1f}x larger). Derating is "
            "legitimate and this does not fail the analysis, but a gap this size "
            "means the vehicle is carrying the mass and power of authority it "
            "never commands — either the derate has a reason worth recording, or "
            "the wheel is the wrong unit."
        )

    floor = analysis.mtq.noise_floor.rate_worst_radps
    inertia_max = float(np.max(vehicle.principal_moments_kgm2))
    if inertia_max * floor > analysis.wheels.usable_momentum_nms:
        out.append(
            f"The B-dot noise floor ({np.degrees(floor):.2f} deg/s) and the usable "
            f"wheel envelope ({analysis.wheels.usable_momentum_nms:g} N.m.s) are "
            "incompatible: the body momentum at the lowest rate B-dot can certify "
            f"({inertia_max * floor:.3g} N.m.s) already exceeds what the wheels may "
            "hold, so no detumble exit threshold satisfies both bounds. The levers "
            "are a quieter magnetometer, a longer B-dot differencing interval "
            "(bounded by BdotMaxSampleDtSec), or a larger certified envelope."
        )

    ellipsoid = analysis.wheels.momentum.ellipsoid_inscribed
    if ellipsoid < analysis.wheels.momentum.inscribed:
        out.append(
            "The wheel array flies the L-infinity allocator (AllocMethodSel = 1), "
            f"which reaches the zonotope ({analysis.wheels.momentum.inscribed:.3g} "
            f"N.m.s guaranteed). An L2 (minimum-norm) allocator under the same "
            f"per-wheel limit would reach only the inscribed ellipsoid "
            f"({ellipsoid:.3g} N.m.s), so changing the allocation method changes "
            "the sizing answer."
        )
    return tuple(out)


def sizing_report(
    vehicle: Vehicle,
    config_path: str | Path,
    assumptions: SizingAssumptions | None = None,
    analysis: SizingAnalysis | None = None,
) -> AnalysisReport:
    """The ADCS sizing analysis as one structured, renderable report.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    config_path : str or pathlib.Path
        The config it was loaded from; recorded as provenance.
    assumptions : SizingAssumptions, optional
        The assumptions in force.
    analysis : SizingAnalysis, optional
        A precomputed analysis, to avoid running it twice when the caller has
        one already.

    Returns
    -------
    analysis.common.report.AnalysisReport
    """
    assumptions = assumptions or SizingAssumptions()
    analysis = analysis or sizing_analysis(vehicle, assumptions)
    budget = analysis.budget
    wheel = analysis.wheels
    return AnalysisReport(
        title=f"ADCS actuator sizing — {vehicle.name}",
        config_path=str(config_path),
        provenance={
            "wheels": (
                f"{wheel.momentum.n_actuators} at "
                f"{vehicle.wheel_max_momentum_nms:g} N.m.s / "
                f"{vehicle.wheel_max_torque_nm:g} N.m; zonotope r_in "
                f"{wheel.momentum.inscribed:.4g}, r_out "
                f"{wheel.momentum.circumscribed:.4g}, ellipsoid "
                f"{wheel.momentum.ellipsoid_inscribed:.4g} N.m.s"
            ),
            "usable": (
                f"{wheel.usable_momentum_nms:.4g} N.m.s = "
                + (
                    "MomentumEnvelopeNms (the flight ceiling binds)"
                    if wheel.envelope_limited
                    else "the wheel zonotope (the hardware binds)"
                )
            ),
            "rods": (
                f"{analysis.mtq.dipole.n_actuators} at "
                f"{vehicle.mtq_max_dipole_am2:g} A.m^2, duty "
                f"{vehicle.mtq_duty_factor:g}; guaranteed dipole "
                f"{analysis.mtq.dipole.inscribed:g} A.m^2, average torque "
                f"{analysis.mtq.average_torque_nm:.3g} N.m"
            ),
            "inertia": (
                "diag("
                + ", ".join(f"{j:g}" for j in vehicle.principal_moments_kgm2)
                + ") kg.m^2, mass "
                + f"{vehicle.mass_kg:g} kg"
            ),
            "orbit": (
                f"a = {vehicle.orbit.sma_m / 1000.0:.1f} km, i = "
                f"{np.degrees(vehicle.orbit.inc_rad):.2f} deg, T = "
                f"{vehicle.orbit.period_s:.0f} s; |B| "
                f"{budget.field.min_t * 1e6:.1f}-{budget.field.max_t * 1e6:.1f} uT"
            ),
            "disturbance": (
                f"total {budget.total_nm:.3g} N.m = secular "
                f"{budget.secular_nm:.3g} + cyclic {budget.cyclic_nm:.3g}"
            ),
            "bdot floor": (
                f"{np.degrees(analysis.mtq.noise_floor.rate_worst_radps):.2f} deg/s "
                f"at |B|_min, "
                f"{np.degrees(analysis.mtq.noise_floor.rate_mean_radps):.2f} deg/s "
                "at |B|_mean"
            ),
        },
        assumptions=assumptions.describe(vehicle.orbit.period_s),
        criteria=analysis.criteria(),
        warnings=_warnings(analysis),
    )


def format_derived(analysis: SizingAnalysis) -> str:
    """Render the derived tuning and its justifications as plain text.

    Separate from :meth:`AnalysisReport.format_text` because these are not
    pass/fail criteria — they are recommendations with arguments attached, and
    forcing them into a criteria table would lose the argument, which is the
    part worth reading.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.

    Returns
    -------
    str
        No trailing newline.
    """
    lines = ["Derived flight parameters (recommendation, committed, and why)", "=" * 70]
    for p in analysis.derived:
        agreement = (
            "no committed value"
            if np.isnan(p.committed)
            else f"committed {p.committed:.4g} = {p.ratio:.3g}x the derived value"
        )
        lines += [
            "",
            f"{p.name}  [{p.units}]",
            f"  derived   : {p.derived:.4g}   ({agreement})",
            f"  formula   : {p.formula}",
            f"  inputs    : {p.inputs}",
            f"  reasoning : {p.reasoning}",
        ]
    return "\n".join(lines)


def format_budget(analysis: SizingAnalysis) -> str:
    """Render the disturbance-torque budget with its secular/cyclic split.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.

    Returns
    -------
    str
        No trailing newline.
    """
    budget = analysis.budget
    header = f"{'term':<28}{'torque [N.m]':>14}{'secular':>14}{'cyclic':>14}   formula"
    lines = [
        "Disturbance-torque budget (worst case, analytic)",
        "=" * 70,
        header,
        "-" * 100,
    ]
    for term in budget.terms:
        lines.append(
            f"{term.name:<28}{term.torque_nm:>14.4e}{term.secular_nm:>14.4e}"
            f"{term.cyclic_nm:>14.4e}   {term.formula}"
        )
        lines.append(f"{'':<28}{term.inputs}")
    lines.append("-" * 100)
    lines.append(
        f"{'total':<28}{budget.total_nm:>14.4e}{budget.secular_nm:>14.4e}"
        f"{budget.cyclic_nm:>14.4e}   (summed, not RSS: worst cases can coincide)"
    )
    return "\n".join(lines)
