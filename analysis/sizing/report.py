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


@dataclass(frozen=True)
class SpecItem:
    """One provenance quantity: what it is, its symbol, its value, its units.

    Provenance used to be written straight into the report as one crammed string
    per group (``"4 at 0.03 N.m.s / 0.002 N.m; zonotope r_in 0.04899, ..."``),
    which a terminal tolerates and a reader does not. The quantities are carried
    apart instead, so the HTML page can set one per line with its symbol typeset
    and its number in the tabular face, and :func:`_provenance` can still join
    them back into the single line the console report wants. One source, two
    renderings.

    Attributes
    ----------
    label : str
        What the quantity is, in words, capitalised.
    symbol : str
        Its symbol as the report's plain-text convention writes it (``r_in``,
        ``|B|_min``), or empty when it has none. Typeset at the presentation
        boundary; never parsed.
    value : str
        The formatted value, or a short phrase for a non-numeric fact.
    units : str
        Display units in the report's dotted convention (``N.m.s``), or empty.
    value_tex : str
        The value as LaTeX, for the one quantity that is not a scalar: the
        inertia tensor, which the page sets as a matrix. Empty everywhere else,
        and the console never sees it — :meth:`one_line` renders
        :attr:`value` exactly as before.
    si : float or None
        The same quantity as a bare SI number, when it is one. Carried beside
        the formatted :attr:`value` so a rendering that wants to choose its own
        SI prefix (the page does; see
        :func:`analysis.sizing.mathfmt.unit_scale`) has the number rather than a
        string to re-parse. ``None`` for a count, a matrix or an identifier —
        anything a prefix would be meaningless on.
    note : str
        What the value means, when that needs a sentence. It is rendered as a
        caption beneath the group, **never inside the value cell**: a value cell
        holds a number, a unit or an identifier, and prose that leaks into one
        makes a table unreadable and unsortable.
    """

    label: str
    symbol: str
    value: str
    units: str
    value_tex: str = ""
    si: float | None = None
    note: str = ""

    def one_line(self) -> str:
        """The item as the console report writes it: label, symbol, value, units."""
        head = f"{self.label} {self.symbol}".strip()
        tail = f"{self.value} {self.units}".strip()
        line = f"{head} {tail}".strip()
        return f"{line} ({self.note})" if self.note else line


def _inertia_item(tensor: np.ndarray) -> SpecItem:
    """The inertia tensor as one provenance quantity, matrix and all.

    Every per-axis result in this report — and every margin
    :mod:`analysis.control` contributes to it — is valid *because* the products
    of inertia are zero. The page therefore shows the whole 3×3 rather than
    asserting the diagonality in prose, which is why the LaTeX is a full
    ``bmatrix`` with the off-diagonal zeros visible.

    :func:`analysis.control.vehicle.load_vehicle` refuses a tensor with non-zero
    products, so on any config that reaches here they are zero. The rendering
    still reads them out of the tensor rather than assuming: a value that is
    displayed and a value that is used must be the same value, and the failure
    mode of assuming is a page that quietly shows zeros a config does not carry.
    The console form stays ``diag(...)`` while they are zero and spells the rows
    out when they are not, so the one-line record cannot become a false claim.

    Parameters
    ----------
    tensor : numpy.ndarray
        Body-frame inertia tensor, shape ``(3, 3)`` [kg·m²].

    Returns
    -------
    SpecItem
    """
    rows = [[float(tensor[i][j]) for j in range(3)] for i in range(3)]
    diagonal = all(rows[i][j] == 0.0 for i in range(3) for j in range(3) if i != j)
    value = (
        "diag(" + ", ".join(f"{rows[i][i]:g}" for i in range(3)) + ")"
        if diagonal
        else "; ".join(", ".join(f"{v:g}" for v in row) for row in rows)
    )
    body = r" \\ ".join(" & ".join(f"{v:g}" for v in row) for row in rows)
    return SpecItem(
        "Inertia tensor",
        "J",
        value,
        "kg.m^2",
        value_tex=rf"J = \begin{{bmatrix}} {body} \end{{bmatrix}}",
    )


def spec_groups(
    analysis: SizingAnalysis,
) -> tuple[tuple[str, tuple[SpecItem, ...]], ...]:
    """The configuration this analysis describes, quantity by quantity.

    The provenance strip of the report, structured. Keys match the flat
    :attr:`~analysis.common.report.AnalysisReport.provenance` dictionary
    :func:`sizing_report` builds from this, so the two renderings cannot drift.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.

    Returns
    -------
    tuple
        ``(key, items)`` pairs in document order.
    """
    vehicle = analysis.vehicle
    wheel = analysis.wheels
    mtq = analysis.mtq
    budget = analysis.budget
    floor = mtq.noise_floor
    # The identifier alone in the value; what binding *means* is a sentence and
    # belongs in the group's caption, where a sentence can be read.
    binding, binding_note = (
        ("MomentumEnvelopeNms", "the certified flight ceiling binds, not the hardware")
        if wheel.envelope_limited
        else ("r_in", "the wheel zonotope binds: the hardware is the smaller limit")
    )
    return (
        (
            "wheels",
            (
                SpecItem("Units installed", "", f"{wheel.momentum.n_actuators}", ""),
                SpecItem(
                    "Momentum per wheel",
                    "",
                    f"{vehicle.wheel_max_momentum_nms:g}",
                    "N.m.s",
                    si=vehicle.wheel_max_momentum_nms,
                ),
                SpecItem(
                    "Torque per wheel",
                    "",
                    f"{vehicle.wheel_max_torque_nm:g}",
                    "N.m",
                    si=vehicle.wheel_max_torque_nm,
                ),
                SpecItem(
                    "Zonotope inscribed radius",
                    "r_in",
                    f"{wheel.momentum.inscribed:.4g}",
                    "N.m.s",
                    si=wheel.momentum.inscribed,
                ),
                SpecItem(
                    "Zonotope circumscribed radius",
                    "r_out",
                    f"{wheel.momentum.circumscribed:.4g}",
                    "N.m.s",
                    si=wheel.momentum.circumscribed,
                ),
                SpecItem(
                    "L2 ellipsoid radius",
                    "",
                    f"{wheel.momentum.ellipsoid_inscribed:.4g}",
                    "N.m.s",
                    si=wheel.momentum.ellipsoid_inscribed,
                ),
            ),
        ),
        (
            "usable",
            (
                SpecItem(
                    "Usable momentum",
                    "",
                    f"{wheel.usable_momentum_nms:.4g}",
                    "N.m.s",
                    si=wheel.usable_momentum_nms,
                ),
                SpecItem("Binding limit", "", binding, "", note=binding_note),
            ),
        ),
        (
            "rods",
            (
                SpecItem("Units installed", "", f"{mtq.dipole.n_actuators}", ""),
                SpecItem(
                    "Dipole per rod",
                    "",
                    f"{vehicle.mtq_max_dipole_am2:g}",
                    "A.m^2",
                    si=vehicle.mtq_max_dipole_am2,
                ),
                SpecItem("Duty factor", "", f"{vehicle.mtq_duty_factor:g}", ""),
                SpecItem(
                    "Guaranteed dipole",
                    "m_in",
                    f"{mtq.dipole.inscribed:g}",
                    "A.m^2",
                    si=mtq.dipole.inscribed,
                ),
                SpecItem(
                    "Average torque",
                    "",
                    f"{mtq.average_torque_nm:.3g}",
                    "N.m",
                    si=mtq.average_torque_nm,
                ),
            ),
        ),
        (
            "inertia",
            (
                _inertia_item(vehicle.inertia_kgm2),
                SpecItem("Mass", "", f"{vehicle.mass_kg:g}", "kg"),
            ),
        ),
        (
            "orbit",
            (
                SpecItem(
                    "Semi-major axis", "a", f"{vehicle.orbit.sma_m / 1000.0:.1f}", "km"
                ),
                SpecItem(
                    "Inclination",
                    "i",
                    f"{np.degrees(vehicle.orbit.inc_rad):.2f}",
                    "deg",
                ),
                SpecItem("Period", "T", f"{vehicle.orbit.period_s:.0f}", "s"),
                SpecItem(
                    "Weakest field", "|B|_min", f"{budget.field.min_t * 1e6:.1f}", "uT"
                ),
                SpecItem(
                    "Strongest field",
                    "|B|_max",
                    f"{budget.field.max_t * 1e6:.1f}",
                    "uT",
                ),
            ),
        ),
        (
            "disturbance",
            (
                SpecItem(
                    "Total torque",
                    "",
                    f"{budget.total_nm:.3g}",
                    "N.m",
                    si=budget.total_nm,
                ),
                SpecItem(
                    "Secular",
                    "",
                    f"{budget.secular_nm:.3g}",
                    "N.m",
                    si=budget.secular_nm,
                ),
                SpecItem(
                    "Cyclic", "", f"{budget.cyclic_nm:.3g}", "N.m", si=budget.cyclic_nm
                ),
            ),
        ),
        (
            "bdot floor",
            (
                SpecItem(
                    "At the weakest field",
                    "|B|_min",
                    f"{np.degrees(floor.rate_worst_radps):.2f}",
                    "deg/s",
                ),
                SpecItem(
                    "At the mean field",
                    "|B|_mean",
                    f"{np.degrees(floor.rate_mean_radps):.2f}",
                    "deg/s",
                ),
            ),
        ),
    )


def _provenance(analysis: SizingAnalysis) -> dict[str, str]:
    """:func:`spec_groups` flattened to the one-line-per-group console form."""
    return {
        key: "; ".join(item.one_line() for item in items)
        for key, items in spec_groups(analysis)
    }


def _warnings(analysis: SizingAnalysis) -> tuple[str, ...]:
    """Conditions that qualify the analysis without failing it."""
    vehicle = analysis.vehicle
    out = [
        "No requirement in the baseline is written on actuator sizing, so the "
        "sizing criteria carry no requirement ID (the derived-parameter check "
        "against the REQ-ACTL-009 envelope is the one exception, and that is a "
        "behaviour requirement). REQ-ACTL-009 governs the "
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
    return AnalysisReport(
        title=f"ADCS actuator sizing — {vehicle.name}",
        config_path=str(config_path),
        provenance=_provenance(analysis),
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
