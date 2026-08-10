"""Structured pass/fail results and their human-readable rendering.

The standing convention (``analysis/CLAUDE.md``): an analysis that has a
pass/fail criterion produces **both** a machine-readable result — which is what
tests assert on, never a parsed string — and a plain-text rendering written
beside the figures, carrying enough context that the table is self-contained:
the requirement each criterion belongs to, the threshold, the measured value,
the margin in absolute and percentage terms, the verdict, the configuration the
numbers came from, and the assumptions in force.

Why margin is a first-class field
---------------------------------
A pass with 2% of margin and a pass with 150% are different engineering
situations and identical booleans. The repo's requirement baseline already
demands margin reporting rather than pass/fail alone (design doc §22.2), and a
margin that is only printed is a number rather than a property — so it is
computed here, once, in the direction the criterion runs.

Units
-----
Whatever the criterion declares. SI internally is a rule about the *analysis*;
a report is a presentation boundary, so dB, degrees and percent belong here.

References
----------
Design doc §13 (analysis), §22.2 (requirements and margin reporting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

#: Criterion senses. ``"min"`` = measured must be at or above the threshold
#: (a margin, a rank ratio); ``"max"`` = at or below it (a sensitivity peak).
SENSES = ("min", "max")


@dataclass(frozen=True)
class Criterion:
    """One threshold, one measurement, one verdict.

    Attributes
    ----------
    name : str
        What was measured, e.g. ``"phase margin (axis x)"``.
    requirement : str
        Requirement ID it belongs to, e.g. ``"REQ-ACTL-006"``. Empty for a
        diagnostic that no requirement is written on.
    threshold : float
        The requirement value, in :attr:`units`.
    measured : float
        The measured value, same units.
    units : str
        Display units, e.g. ``"dB"``, ``"deg"``, ``"-"``.
    sense : str
        One of :data:`SENSES`.
    note : str
        Optional one-line context carried into the rendering.
    formula : str
        The closed form :attr:`note` opens with, when it opens with one, exactly
        as it is spelled there (``"tau_secular * T_desat"``). Empty when the note
        is prose. Carried so a rendering can lift the equation out of the
        sentence without pattern-matching its own output back into structure.
    formula_tex : str
        The same formula as LaTeX, written beside it at its source. Empty means
        the formula has no typeset form and must be set as plain text — never
        guessed at. Presentation only; no verdict depends on it.
    """

    name: str
    requirement: str
    threshold: float
    measured: float
    units: str
    sense: str = "min"
    note: str = ""
    formula: str = ""
    formula_tex: str = ""

    def __post_init__(self) -> None:
        if self.sense not in SENSES:
            raise ValueError(f"sense must be one of {SENSES}, got {self.sense!r}")

    @property
    def passes(self) -> bool:
        """The measurement satisfies the threshold in the declared sense."""
        if math.isnan(self.measured):
            return False
        return (
            self.measured >= self.threshold
            if self.sense == "min"
            else self.measured <= self.threshold
        )

    @property
    def margin(self) -> float:
        """Signed margin in :attr:`units`; positive means inside the requirement."""
        if math.isnan(self.measured):
            return float("nan")
        return (
            self.measured - self.threshold
            if self.sense == "min"
            else self.threshold - self.measured
        )

    @property
    def margin_pct(self) -> float:
        """Margin as a percentage of the threshold [%].

        ``inf`` when the measurement is unbounded (an infinite gain margin is a
        real and reportable result), ``nan`` when the threshold is zero and the
        percentage has no meaning.
        """
        if self.threshold == 0.0:
            return float("nan")
        return 100.0 * self.margin / abs(self.threshold)


@dataclass(frozen=True)
class AnalysisReport:
    """A complete analysis result: criteria, provenance and assumptions.

    Attributes
    ----------
    title : str
        What was analysed.
    config_path : str
        The configuration file every number was derived from — provenance, so a
        report read six months later says which vehicle it describes.
    provenance : dict
        Extra key/value context, e.g. the gains, the sample period, the analysis
        mode. Rendered verbatim.
    assumptions : tuple of str
        The modelling assumptions in force, one per line. A margin without its
        assumptions is not a result.
    criteria : tuple of Criterion
        Every pass/fail check, in report order.
    warnings : tuple of str
        Conditions that do not fail the analysis but qualify it — the SISO
        validity boundary being the motivating case.
    """

    title: str
    config_path: str
    provenance: dict[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        """Every criterion passes. Warnings do not fail a report."""
        return all(c.passes for c in self.criteria)

    def failures(self) -> list[Criterion]:
        """The criteria that did not pass, in report order.

        Returns
        -------
        list of Criterion
            Empty when :attr:`passes`.
        """
        return [c for c in self.criteria if not c.passes]

    def by_requirement(self, requirement: str) -> list[Criterion]:
        """Criteria belonging to one requirement ID.

        Parameters
        ----------
        requirement : str
            Requirement ID, e.g. ``"REQ-ACTL-006"``.

        Returns
        -------
        list of Criterion
        """
        return [c for c in self.criteria if c.requirement == requirement]

    def format_text(self) -> str:
        """Render as a self-contained plain-text report.

        Returns
        -------
        str
            Title, provenance, assumptions, the criteria table, any warnings,
            and the overall verdict. No trailing newline.
        """
        verdict = "PASS" if self.passes else "FAIL"
        out = [
            f"{self.title} — {verdict}",
            "=" * max(len(self.title) + 8, 60),
            f"config      : {self.config_path}",
        ]
        out += [f"{k:<12}: {v}" for k, v in self.provenance.items()]
        if self.assumptions:
            out += ["", "Assumptions in force:"]
            out += [f"  - {a}" for a in self.assumptions]
        out += ["", _table(self.criteria)]
        if self.warnings:
            out += ["", "Warnings (do not fail the analysis, but qualify it):"]
            out += [f"  ! {w}" for w in self.warnings]
        out += ["", f"Overall: {verdict} ({len(self.failures())} failing criteria)"]
        return "\n".join(out)

    def write_text(self, path: str | Path) -> Path:
        """Write :meth:`format_text` to @p path, creating parent directories.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.format_text() + "\n")
        return target


def _fmt(value: float) -> str:
    """Fixed-width number that stays readable across nine orders of magnitude."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e5):
        return f"{value:.3e}"
    return f"{value:.3f}"


def _table(criteria: tuple[Criterion, ...]) -> str:
    """The criteria table: requirement, threshold, measured, margin, verdict."""
    header = (
        f"{'requirement':<14}{'criterion':<52}{'threshold':>20}{'measured':>12}"
        f"{'margin':>12}{'margin %':>11}  {'verdict':<7}"
    )
    rows = [header, "-" * len(header)]
    for c in criteria:
        sense = "≥" if c.sense == "min" else "≤"
        rows.append(
            f"{c.requirement or '-':<14}{c.name:<52}"
            f"{sense + ' ' + _fmt(c.threshold) + ' ' + c.units:>20}"
            f"{_fmt(c.measured):>12}{_fmt(c.margin):>12}{_fmt(c.margin_pct):>11}  "
            f"{'PASS' if c.passes else 'FAIL':<7}"
        )
        if c.note:
            rows.append(f"{'':<16}{c.note}")
    return "\n".join(rows)
