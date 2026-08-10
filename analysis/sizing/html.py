"""The sizing report as one self-contained, interactive HTML page.

The console report (:meth:`analysis.common.report.AnalysisReport.format_text`)
remains the record the tests assert on and is unchanged; this module is a second
*rendering* of the same structured objects, for the case the text serves badly:
a design review, where the headline is a three-dimensional achievable set a
reader has to rotate to believe.

Nothing here computes a verdict. Every number and every PASS/FAIL word comes
from the :class:`~analysis.common.report.AnalysisReport` and the
:class:`~analysis.sizing.report.SizingAnalysis` handed in.

A document, not a dump
----------------------
The page is laid out as an engineering document: a sticky header carrying the
vehicle and the provenance; a section nav; and the criteria grouped by the
family they judge (wheel momentum, wheel torque, magnetorquer authority, control
tuning) rather than one flat list, led inside each group by the tightest margin.
Margins are shown in absolute and percentage terms **in the same cell**, because
a pass with 2 % of margin and a pass with 150 % are different engineering
situations.

The hero is the thesis
----------------------
The page opens with one sentence saying what the analysis concluded, and
directly beneath it the 3D momentum envelope, which is that sentence made
visible: requirement vectors drawn inside the nested capability surfaces. The
vehicle summary follows as a compact spec strip rather than leading, because a
reviewer arrives wanting the answer and reads the configuration only once the
answer is in hand. On a failure the sentence names the count and the worst
criterion, since "FAIL" on its own sends the reader hunting through a table.

Self-explaining, not self-evident
---------------------------------
The report's short codes are its own: D1 through D4 name the wheel momentum
drivers, M1 through M3 the magnetorquer criteria, and a reader meeting them for
the first time cannot expand either from the row. Two things fix that, and both
are needed: the code is expanded **in the criterion label itself**
(:func:`analysis.sizing.mathfmt.criterion_label` sets ``D1b · Post-B-dot
handover``), so the code is never the only identifier a row carries; and each
family opens with a **visible** one-line definition per code
(:func:`_glossary`), said once per family rather than repeated on every row. It
was a disclosure and stayed shut, which is the same as not being there. The
criteria section opens with :data:`_HOW_TO_READ_A_ROW` —
what threshold, measured and margin mean for a minimum-sense criterion against a
maximum-sense one, and which rows the 30 % convention actually governs. One row,
the commanded wheel torque against the installed unit, is *designed* to land on
its threshold at 0 % margin, and says so in its own note rather than leaving the
reader to read a zero as a near miss.

Prefixed units, chosen per family
---------------------------------
``0.0072 N·m·s`` is a number a reviewer has to count zeros in. Every numeric
block on the page therefore picks one SI prefix for each units string it carries
(:func:`analysis.sizing.mathfmt.unit_scale`) and sets every value in that block
through it, so the figure reads ``7.2 mN·m·s`` and the threshold it is compared
against reads in the same unit. Per family and never per cell: a column whose
cells each chose their own prefix would be unreadable and would invite exactly
the mis-comparison the prefix was meant to prevent. The scaling is presentation
only; the report objects and the plain-text rendering keep SI base units.

The provenance strip is a table, not a string
---------------------------------------------
:func:`analysis.sizing.report.spec_groups` carries each configuration quantity
apart — label, symbol, value, units — so the page sets one per line with the
symbol typeset and the number in the tabular face. The console gets the same
quantities joined into its one-line-per-group form, from the same source, so the
two cannot drift.

Scannable by default, complete on demand
----------------------------------------
A reviewer reads a criteria table to find the row that surprises them, not to
read prose. So each row, card, caption and warning shows **one sentence** plus
its numbers, and everything longer collapses into a ``<details>`` disclosure.
Nothing is deleted: the justifications are the reason this report exists, they
are simply one click away rather than in the way of the next row.

The console strings are set for a fixed-width terminal, so they are typeset on
the way in. Formulae go through :mod:`analysis.sizing.texmath`, which typesets
the LaTeX **the report objects carry themselves** with KaTeX; everything else
goes through :mod:`analysis.sizing.mathfmt`, which sets ``N.m.s`` as ``N·m·s``
and is also what a formula with no LaTeX degrades to — plainly, in the body
face, never as pseudo-mathematics. Em dashes and ``**emphasis**`` are console
conventions and are dropped here. All of it is presentation-only and lives
entirely in this layer: the report objects and the plain-text rendering keep
their original strings.

Every symbol the page sets is defined once, in the Nomenclature section, beside
its units and — where the two are the same quantity — the committed flight
parameter it corresponds to. That last column is the link a reviewer needs and
the one no formula carries.

Self-contained by construction
------------------------------
plotly.js is inlined into the first figure (``include_plotlyjs="inline"``),
KaTeX's stylesheet, library and eight WOFF2 faces are inlined by
:func:`analysis.sizing.texmath.katex_assets`, and the matplotlib figures are
embedded as ``data:`` URIs. No external stylesheet, no web font, no CDN. The
result is one file that opens offline, survives being emailed, and fetches
nothing at runtime. It is a few megabytes for exactly that reason.

Escaping
--------
Every value that reaches the page goes through :func:`html.escape` — including
before any math substitution, see :func:`analysis.sizing.mathfmt.math_html`.
Vehicle names, config paths and parameter prose are config-derived, i.e.
untrusted input, and a report that renders a spacecraft name into live markup is
a report that can be made to lie.

References
----------
Design doc §12 (analysis tools), §21.2 (generated artifacts);
``analysis/CLAUDE.md`` (the reporting convention).
"""

from __future__ import annotations

import base64
import html as _html
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go

from analysis.common.report import AnalysisReport, Criterion
from analysis.sizing.interactive import (
    disturbance_figure,
    margin_figure,
    momentum_envelope_figure,
    torque_envelope_figure,
)
from analysis.sizing.mathfmt import (
    UnitScale,
    _num,
    criterion_label,
    math_html,
    percent,
    provenance_label,
    sentence_case,
    signed,
    unit_html,
    unit_scale,
)
from analysis.sizing.report import SizingAnalysis, SpecItem, spec_groups
from analysis.sizing.texmath import (
    katex_assets,
    tex_html,
    tex_symbol,
    tex_value,
)


def _esc(value: object) -> str:
    """Escape any value for interpolation into the page."""
    return _html.escape(str(value), quote=True)


#: Em dashes are set for prose read at length; this page is scanned. Each one
#: becomes a comma at render time so the source strings, which the plain-text
#: report shares, are left alone. ``**emphasis**`` is likewise console markup
#: with no meaning here.
_EM_DASH = re.compile(r"\s*—\s*")

#: A sentence boundary: ``.`` followed by whitespace only, so ``0.5 N.m.s`` and
#: ``REQ-ACTL-009`` survive.
_SENTENCE = re.compile(r"(?<=\.)\s+")

#: A clause boundary, used only on a note long enough that one sentence is
#: already a paragraph. The report's notes use ``;`` where a full stop would do.
_CLAUSE = re.compile(r"(?<=;)\s+")

#: Below this, a string is short enough to read whole and is not split [chars].
_LONG = 150

#: Below this, a remainder is not worth a disclosure of its own [chars].
_WORTH_HIDING = 60

#: Clauses repeated on many rows, lifted out and said once in the family
#: subhead instead. Stripped at render time; the console report keeps them,
#: since a terminal has no subhead to carry them.
_BOILERPLATE = ("; capability is min(zonotope r_in, MomentumEnvelopeNms)",)


def _plain(text: object) -> str:
    """A source string as the page sets prose: no em dashes, no ``**``."""
    return _EM_DASH.sub(", ", str(text)).replace("**", "")


def _prose(text: object) -> str:
    """Prose, de-dashed, sentence-cased and set as mathematics."""
    return math_html(sentence_case(_plain(text)))


def _split(text: str) -> tuple[str, str]:
    """What stays in the open, and what collapses beneath it.

    The first sentence leads. A note with no full stop splits at a semicolon
    instead, but only once it is long enough that leaving it whole would be a
    paragraph in a table cell; and a remainder too short to be worth a click
    stays where it is.
    """
    text = text.strip()
    parts = _SENTENCE.split(text, maxsplit=1)
    if len(parts) == 1 and len(text) > _LONG:
        parts = _CLAUSE.split(text, maxsplit=1)
    if len(parts) == 1 or len(parts[1]) < _WORTH_HIDING:
        return text, ""
    # The semicolon it was split at would otherwise dangle at the end of the
    # visible line, pointing at a clause that is now behind a disclosure.
    return parts[0].rstrip("; "), parts[1]


def _why(body: str, label: str = "Why") -> str:
    """The reasoning, one click away: what shows by default is the verdict."""
    if not body.strip():
        return ""
    return f"<details><summary>{_esc(label)}</summary><div>{body}</div></details>"


def _lead_and_why(text: object, label: str = "Why") -> str:
    """One sentence in the open, the remainder collapsed beneath it."""
    head, tail = _split(_plain(text))
    return f'<div class="lead">{_prose(head)}</div>' + _why(_prose(tail), label)


def _unit(units: object) -> str:
    """A units string, set as markup and styled secondary to its number.

    ``×`` is a multiplier rather than a unit and sets tight against its number
    (``10×``); everything else takes the usual space (``0.38 N·m·s``).
    """
    rendered = unit_html(units)
    if not rendered:
        return ""
    separator = "" if rendered == "×" else " "
    return f'{separator}<span class="u">{rendered}</span>'


def _math(text: object) -> str:
    """A formula or an input list, set as mathematics."""
    return f'<span class="m">{math_html(text)}</span>'


# --------------------------------------------------------------------------
# Criterion families
# --------------------------------------------------------------------------

#: How the flat criterion list is grouped for the reader, in document order —
#: which follows the report's own narrative, wheels → rods → tuning. The
#: keyword sets are disjoint over the current criteria, so this is a display
#: order and not a priority; anything a future criterion introduces falls into
#: the last family rather than being dropped.
_FAMILIES: tuple[tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        "Wheel momentum",
        "Can the array hold what the sizing drivers accumulate? "
        "Usable capability is min(zonotope r_in, MomentumEnvelopeNms).",
        ("D1", "D2", "D3", "oversizing", "handover"),
        (
            (
                "D1 · Raw tip-off absorption",
                "Momentum if the wheels catch the separation tumble unaided.",
            ),
            (
                "D1b · Post-B-dot handover",
                "Momentum at the rate B-dot actually hands over at "
                "(DetumbleExitRadps).",
            ),
            (
                "D2 · Cyclic storage",
                "Momentum stored over a quarter orbit by a disturbance that "
                "reverses.",
            ),
            (
                "D3 · Secular accumulation",
                "Momentum built between desaturations by the non-reversing "
                "disturbance.",
            ),
            (
                "D4 · Slew agility",
                "Momentum at the commanded slew rate; judged only when the "
                "config gives one.",
            ),
        ),
    ),
    (
        "Wheel torque",
        "Is there control torque in every direction, disturbances included?",
        ("wheel torque", "commanded wheel torque"),
        (),
    ),
    (
        "Magnetorquer authority",
        "Can the rods desaturate the wheels and detumble the vehicle?",
        ("M1 ", "M2 ", "M3 ", "M4 "),
        (
            (
                "M1 · Desaturation authority",
                "Rod torque must beat the secular disturbance, or the wheels "
                "saturate whatever their size.",
            ),
            (
                "M2 · Detumble authority",
                "Momentum the rods can remove inside the detumble budget.",
            ),
            (
                "M3 · Exit threshold",
                "The committed exit rate against the B-dot measurement noise "
                "floor; below it, B-dot commands on noise.",
            ),
            (
                "M4 · Saturated convergence",
                "The largest rate B-dot can remove at all once the rods rail. "
                "M2 is an impulse bound and does not see this: a design can "
                "pass M2 and still never converge.",
            ),
        ),
    ),
    (
        "Control tuning and thresholds",
        "Are the committed flight parameters self-consistent and in bounds?",
        (),
        (),
    ),
)


def _family_of(criterion: Criterion) -> int:
    """Index into :data:`_FAMILIES`; the last family is the catch-all."""
    for index, (_, _, keywords, _) in enumerate(_FAMILIES):
        if any(keyword in criterion.name for keyword in keywords):
            return index
    return len(_FAMILIES) - 1


def _worst_first(criterion: Criterion) -> float:
    """Sort key: the tightest margin leads. A non-finite margin sorts last."""
    value = criterion.margin_pct
    return value if math.isfinite(value) else math.inf


def _document_order(report: AnalysisReport) -> list[list[Criterion]]:
    """The criteria as the document presents them: by family, worst margin first.

    One ordering, computed once and used by both the table and the margin chart,
    so a reader meets the criteria in the same sequence in both. Sorting inside a
    family and never across it keeps the grouping meaningful; leading with the
    tightest margin puts the row worth arguing about at the top of its group.
    """
    grouped: list[list[Criterion]] = [[] for _ in _FAMILIES]
    for criterion in report.criteria:
        grouped[_family_of(criterion)].append(criterion)
    return [sorted(family, key=_worst_first) for family in grouped]


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

_CSS = """
:root {
  /* Declared, and not merely implied by the palette below: a browser with
     automatic dark mode enabled will otherwise invert a page that states no
     scheme, which turns this document's warm paper into a dark surface with
     figures that were drawn for the light one. Saying "light" opts out. */
  color-scheme: light;

  /* Surface and ink. Warm off-white paper, near-black ink: an engineering memo,
     not an application chrome. */
  --surface: #fcfcfb; --panel: #ffffff; --panel-2: #f7f7f5;
  --ink: #1a1d21; --ink-2: #5b6470; --muted: #5b6470;
  --line: #e4e2dd; --line-2: #eeece8;

  /* Structure, never data. The one brand hue on the page: section eyebrows,
     table heads, rules, the spine of a card. Nothing measured is ever drawn in
     it, so a slate mark on this page always means "this is the frame". */
  --slate: #24364a; --slate-soft: #eef1f5;

  /* Status. Always accompanied by the word PASS or FAIL, per the standing
     convention. The pair is separable under protanopia and deuteranopia; a
     warning is deliberately NOT amber, because amber against this red is the
     one pairing that collapses under protan simulation at any lightness the
     contrast rules allow. Warnings are neutral slate plus a triangle glyph. */
  --pass: #0d5226; --fail: #e5484d;
  --pass-bg: #eaf2ec; --fail-bg: #fdeeee;
  --warn: #445062; --warn-bg: #f4f5f7;

  /* Categorical, for figures and for anything on the page that keys to them.
     Two, assigned in a fixed order and never cycled. */
  --series-1: #0969da; --series-2: #bc4c00;

  --display: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
    sans-serif;
  --body: Charter, "Bitstream Charter", "Iowan Old Style", Georgia,
    "Times New Roman", serif;
  --mono: ui-monospace, "Cascadia Mono", SFMono-Regular, Menlo, Consolas,
    monospace;
  --shadow: 0 1px 2px rgba(26, 29, 33, .05);
}
/* Light unconditionally: there is deliberately no prefers-color-scheme
   override. This page is a design-review artifact: two reviewers reading the
   same file must see the same document, and an appearance that changes with the
   reader's OS setting is a liability, not a feature. The plotly figures are set
   to the same light palette in analysis/sizing/interactive.py. */
* { box-sizing: border-box; }
html { scroll-padding-top: 7.5rem; }
@media (prefers-reduced-motion: no-preference) {
  html { scroll-behavior: smooth; }
}
body {
  /* Explicit, not inherited: a transparent body borrows whatever ground the
     viewer paints behind it, which on a dark browser is the failure this page
     exists to avoid. */
  margin: 0; padding: 0; background: var(--surface); color: var(--ink);
  font: 16px/1.65 var(--body);
  -webkit-font-smoothing: antialiased;
}
/* Every number, unit and identifier on the page is set in the mono face with
   tabular figures, so a column of them aligns and a parameter name never reads
   as prose. */
.num, td.n, th.n, .u, .pct, .req, .compare .v, .spec dd, .tally, .topbar .meta {
  font-family: var(--mono); font-variant-numeric: tabular-nums;
}
main { max-width: 1180px; margin: 0 auto; padding: 0 1.5rem 6rem; }

/* ---- sticky document header ---- */
.topbar {
  position: sticky; top: 0; z-index: 20;
  background: var(--panel); border-bottom: 1px solid var(--line);
}
.topbar .row {
  max-width: 1180px; margin: 0 auto; padding: .6rem 1.5rem;
  display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
}
.topbar .who { display: flex; align-items: baseline; gap: .7rem; min-width: 0; }
.topbar .craft {
  font-family: var(--display); font-size: 1.05rem; font-weight: 650;
  letter-spacing: -.015em;
}
.topbar .what {
  font-family: var(--display); color: var(--muted); font-size: .72rem;
  font-weight: 650; letter-spacing: .1em; text-transform: uppercase;
}
.topbar .meta {
  margin-left: auto; text-align: right; color: var(--muted);
  font-size: .72rem; line-height: 1.4;
}
nav.sections { border-top: 1px solid var(--line-2); overflow-x: auto; }
nav.sections ol {
  max-width: 1180px; margin: 0 auto; padding: 0 1.5rem; list-style: none;
  display: flex; gap: .1rem; white-space: nowrap;
}
nav.sections a {
  display: block; padding: .4rem .7rem; color: var(--muted);
  text-decoration: none; font-family: var(--display); font-size: .74rem;
  font-weight: 650; letter-spacing: .06em; text-transform: uppercase;
  border-bottom: 2px solid transparent;
}
nav.sections a:hover { color: var(--slate); border-bottom-color: var(--slate); }

/* ---- verdict pill ---- */
.pill {
  display: inline-block; padding: .2rem .7rem; border-radius: 2px;
  font-family: var(--display); font-size: .78rem; font-weight: 700;
  letter-spacing: .12em; border: 1px solid currentColor;
}
.pill.pass { color: var(--pass); background: var(--pass-bg); }
.pill.fail { color: var(--fail); background: var(--fail-bg); }

/* ---- the hero: the thesis, then the thesis made visible ---- */
.hero { margin: 2.2rem 0 0; scroll-margin-top: 7.5rem; }
.statement {
  display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap;
  padding-bottom: 1rem; border-bottom: 2px solid var(--slate);
}
.statement .pill { flex: none; }
.statement h1 {
  font-family: var(--display); font-size: 1.32rem; font-weight: 650;
  letter-spacing: -.018em; line-height: 1.3; margin: 0; flex: 1 1 22rem;
  min-width: 0; color: var(--ink);
}
.statement.fail h1 > b { color: var(--fail); font-weight: 650; }
.statement .tally {
  margin-left: auto; color: var(--muted); font-size: .78rem; white-space: nowrap;
}
.hero > figure { margin-top: 0; border: 0; border-radius: 0; box-shadow: none;
  padding: 0; background: none; }
.hero > figure > figcaption { padding: 0; }

/* ---- the spec strip ---- */
.spec { margin: 1.6rem 0 0; border-top: 1px solid var(--line); }
.spec dl {
  margin: 0; display: grid; grid-template-columns: minmax(0, 1fr);
  border-bottom: 1px solid var(--line);
}
@media (min-width: 700px) {
  .spec dl { grid-template-columns: 10rem minmax(0, 1fr); }
}
.spec dt {
  font-family: var(--display); padding: .5rem .1rem .05rem; color: var(--slate);
  font-size: .7rem; font-weight: 650; letter-spacing: .1em;
  text-transform: uppercase;
}
.spec dd { margin: 0; padding: .05rem .1rem .5rem; font-size: .8rem;
  color: var(--ink-2); }
@media (min-width: 700px) {
  .spec dt { padding: .45rem 1rem .45rem .1rem;
    border-top: 1px solid var(--line-2); }
  .spec dd { padding: .45rem .1rem; border-top: 1px solid var(--line-2); }
}
.spec dl > dt:first-of-type, .spec dl > dt:first-of-type + dd { border-top: 0; }
/* One quantity per line: what it is, its symbol, its value with units. Packing
   six of these into one string is what the console has to do, not what a
   reviewer should have to read. */
.spec .quantities { display: grid; gap: .1rem .9rem; }
@media (min-width: 700px) {
  .spec .quantities {
    grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr));
  }
}
.spec .q {
  display: grid; grid-template-columns: minmax(0, 1fr) 3.2rem auto;
  align-items: baseline; gap: .5rem; padding: .1rem 0;
}
.spec .ql { font-family: var(--body); color: var(--muted); font-size: .82rem; }
.spec .qs { font-family: var(--body); color: var(--ink-2); font-size: .82rem;
  text-align: right; }
.spec .qv { font-family: var(--mono); font-variant-numeric: tabular-nums;
  color: var(--ink); font-size: .82rem; text-align: right; white-space: nowrap; }
/* The inertia tensor is a matrix, so its row takes the full width of the strip,
   sets its value left of the label rather than flush right where a 3x3 would be
   clipped, and is allowed the height the matrix needs. */
.spec .q.wide {
  grid-column: 1 / -1; align-items: center;
  grid-template-columns: max-content minmax(0, 1fr);
}
.spec .q.wide .qv {
  text-align: left; white-space: normal; font-family: var(--body);
  font-size: 1rem;
}

/* A quantity that needs a sentence gets it here, under the group. A value cell
   holds a number, a unit or an identifier; prose in one is a defect. */
.spec .qnote { margin: .35rem 0 0; font-size: .78rem; color: var(--muted);
  max-width: 60ch; }

/* ---- glossary and the reading guide ---- */
p.xref { margin: .35rem 0 0; font-size: .78rem; color: var(--muted); }
dl.glossary { margin: .1rem 0 0; display: grid; gap: .1rem .8rem; }
@media (min-width: 700px) {
  dl.glossary { grid-template-columns: 15rem minmax(0, 1fr); }
}
dl.glossary dt {
  font-family: var(--mono); font-size: .78rem; color: var(--ink);
  font-weight: 600; padding-top: .25rem;
}
dl.glossary dd {
  margin: 0; font-family: var(--body); font-weight: 400; letter-spacing: 0;
  color: var(--ink-2); font-size: .84rem; padding-bottom: .25rem;
}
details.howto { margin: 0 0 1.1rem; background: var(--panel);
  border: 1px solid var(--line); padding: .55rem .9rem; }
details.howto > div { border-left: 0; padding-left: 0; max-width: 84ch; }
details.howto p { margin: .5rem 0; }

/* ---- sections ---- */
section { margin: 3.6rem 0 0; scroll-margin-top: 7.5rem; }
section > h2 {
  font-family: var(--display);
  font-size: .74rem; font-weight: 650; letter-spacing: .14em;
  text-transform: uppercase; color: var(--slate);
  margin: 0 0 .2rem; padding-bottom: .45rem;
  border-bottom: 1px solid var(--slate);
}
section > .lede {
  color: var(--ink-2); font-size: .95rem; margin: .9rem 0 1.2rem;
  max-width: 80ch;
}
h3 { font-family: var(--display); font-size: .95rem; font-weight: 650;
  letter-spacing: -.01em; margin: 0 0 .5rem; }

/* ---- tables ---- */
.scroll { overflow-x: auto; border: 1px solid var(--line);
  background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
/* Not sticky: the table lives in an overflow-x container, where a sticky
   header positions against that container and floats over the first row. The
   family subheads are what keep the reader oriented instead. */
thead th {
  background: var(--slate); color: #f2f4f7;
  font-family: var(--display);
  font-size: .68rem; font-weight: 650; letter-spacing: .1em;
  text-transform: uppercase; text-align: left; white-space: nowrap;
  padding: .55rem .75rem; cursor: pointer; user-select: none;
}
thead th::after { content: " \\2195"; opacity: .5; font-size: .9em; }
thead th:focus-visible { outline: 2px solid var(--series-1); outline-offset: -3px; }
th.n, td.n { text-align: right; }
td { padding: .5rem .75rem; border-top: 1px solid var(--line-2);
  vertical-align: top; }
td.n { white-space: nowrap; }
tbody.group th {
  font-family: var(--display);
  text-align: left; padding: .9rem .75rem .4rem;
  border-top: 1px solid var(--slate); background: var(--panel-2);
  font-size: .78rem; font-weight: 650; letter-spacing: .02em; color: var(--slate);
}
tbody.group th .why {
  display: block; font-family: var(--body); font-weight: 400; letter-spacing: 0;
  color: var(--muted); font-size: .8rem; margin-top: .15rem;
}
tbody tr:hover td { background: var(--panel-2); }
tr.fail td { background: var(--fail-bg); }
tr.fail:hover td { background: var(--fail-bg); }
tr.fail td:first-child { box-shadow: inset 3px 0 0 var(--fail); }
.crit { font-weight: 600; color: var(--ink); }
.req { font-size: .78rem; color: var(--muted); white-space: nowrap; }
.u { color: var(--muted); font-size: .84em; }
.pct { color: var(--muted); font-size: .84em; }
.verdict { font-family: var(--display); font-weight: 700; font-size: .74rem;
  letter-spacing: .1em; }
.verdict.pass { color: var(--pass); }
.verdict.fail { color: var(--fail); }
.note { color: var(--muted); font-size: .84rem; margin-top: .25rem;
  max-width: 68ch; }
tfoot td { border-top: 1px solid var(--slate); font-weight: 650; }
/* The nomenclature is read, not sorted: its columns carry no order worth
   imposing, so its headings are plain rather than the sortable affordance. */
table.nomen thead th { cursor: default; }
table.nomen thead th::after { content: ""; }
table.nomen td.sym { white-space: nowrap; font-size: 1.02rem; width: 1%; }
table.nomen td.n { color: var(--muted); }

/* ---- mathematics ---- */
/* Two faces, and the difference is meant to be visible. `.tex` carries LaTeX
   from its source and is typeset by KaTeX on load. `.m` is everything else:
   a bare symbol in the provenance strip, which needs no more than a subscript,
   and any formula whose source carries no LaTeX, which is set as italic body
   text rather than converted by guesswork. A reader can tell at a glance which
   is which, and that is the honest state to be in. */
.m { font-family: var(--body); font-style: italic; letter-spacing: .01em; }
.m sub, .m sup { font-style: normal; font-size: .68em; }
/* Until the script runs, and forever if it never does, a `.tex` span holds the
   same Unicode rendering `.m` would have. */
.tex { font-family: var(--body); font-style: italic; letter-spacing: .01em; }
.tex sub, .tex sup { font-style: normal; font-size: .68em; }
.katex { font-style: normal; letter-spacing: normal; }
.formula {
  display: block; background: var(--slate-soft); border-left: 2px solid var(--slate);
  padding: .4rem .65rem; margin: 0; font-size: 1.02rem; color: var(--ink);
  overflow-x: auto; white-space: nowrap;
}
/* KaTeX sets display mode as a centred block; here an equation is a value in a
   document, so it stays on the left of whatever cell holds it. */
.katex-display { margin: .3rem 0; text-align: left; }
.katex-display > .katex { text-align: left; }

/* ---- disclosures ---- */
.lead { margin: 0; }
details { margin: .3rem 0 0; }
details > summary {
  cursor: pointer; color: var(--slate); font-family: var(--display);
  font-size: .76rem; font-weight: 650; letter-spacing: .02em;
  list-style: none; display: inline-flex; align-items: center; gap: .3rem;
  border-radius: 2px;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "\\25B8"; font-size: .8em; }
details[open] > summary::before { content: "\\25BE"; }
details > summary:hover { text-decoration: underline; }
details > summary:focus-visible {
  outline: 2px solid var(--series-1); outline-offset: 2px;
}
details > div {
  margin-top: .4rem; color: var(--ink-2); font-size: .88rem; max-width: 72ch;
  border-left: 2px solid var(--line); padding-left: .8rem;
}

/* ---- figures ---- */
figure { margin: 1.8rem 0; background: var(--panel);
  border: 1px solid var(--line); padding: .8rem; }
figure img { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--ink-2); font-size: .9rem; margin-top: .7rem;
  padding: 0 .3rem; max-width: 84ch; }

/* ---- cards ---- */
.cards { display: grid; gap: 1.1rem;
  grid-template-columns: repeat(auto-fit, minmax(23rem, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line);
  border-top: 2px solid var(--slate); padding: 1rem 1.1rem; }
.card h3 { font-family: var(--mono); font-size: .9rem; letter-spacing: -.01em; }
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: .1rem 1rem;
  margin: .7rem 0 .9rem; padding: .6rem 0;
  border-top: 1px solid var(--line-2); border-bottom: 1px solid var(--line-2); }
.compare .k, .card .k {
  font-family: var(--display); color: var(--muted); font-size: .68rem;
  font-weight: 650; letter-spacing: .1em; text-transform: uppercase;
}
.compare .v { font-size: 1.05rem; font-weight: 600; color: var(--ink); }
.compare .d { grid-column: 1 / -1; color: var(--muted); font-size: .84rem;
  margin-top: .25rem; }
.card .row { margin: .55rem 0; }
.card .k { display: block; margin-bottom: .2rem; }
.card > details { margin-top: .9rem; }

/* ---- prose blocks ---- */
.panel { background: var(--panel); border: 1px solid var(--line);
  padding: 1rem 1.3rem; }
/* A warning is neutral slate and a triangle, never amber: see the palette note
   at the top. The glyph is what a reader in any colour vision sees first. */
.panel.warn { border-left: 3px solid var(--warn); background: var(--warn-bg); }
ul.block { margin: 0; padding-left: 1.2rem; max-width: 84ch; }
ul.block li { margin: .6rem 0; color: var(--ink-2); font-size: .92rem; }
ul.block li::marker { color: var(--muted); }
ul.block.warn { list-style: none; padding-left: 0; }
ul.block.warn > li { padding-left: 1.5rem; position: relative; }
ul.block.warn > li::before {
  content: "\\25B2"; position: absolute; left: 0; top: .15em;
  color: var(--warn); font-size: .7em;
}

/* ---- print ---- */
@media print {
  html { scroll-padding-top: 0; }
  body { background: #fff; color: #000; }
  .topbar { position: static; }
  nav.sections { display: none; }
  thead th { position: static; background: #fff; color: #000;
    border-bottom: 1px solid #000; }
  .scroll, .panel, .card, figure, .spec {
    box-shadow: none; border-color: #999; break-inside: avoid;
  }
  section > h2, .statement { break-after: avoid; }
  tr.fail td { background: #fff; }
  /* Paper cannot be clicked: the collapsed reasoning prints. */
  details > summary { display: none; }
  details > div { display: block; }
}
"""

_JS = """
document.querySelectorAll('table.sortable').forEach(function (table) {
  table.querySelectorAll('thead th').forEach(function (th, index) {
    // Sorting is a control, so it is reachable and operable from the keyboard.
    // Done here rather than in the markup because a page with scripting off
    // cannot sort at all, and a focusable header that does nothing is worse
    // than a plain one.
    th.tabIndex = 0;
    th.setAttribute('role', 'button');
    var sort = function () {
      var asc = th.dataset.asc !== 'true';
      table.querySelectorAll('thead th').forEach(function (o) { o.dataset.asc = ''; });
      th.dataset.asc = asc ? 'true' : 'false';
      // Sort inside each family, never across them: the grouping is the
      // document's structure, not a sort order to be discarded.
      Array.prototype.forEach.call(table.tBodies, function (body) {
        if (body.classList.contains('group')) { return; }
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[index], y = b.cells[index];
          if (!x || !y) { return 0; }
          var nx = parseFloat(x.dataset.sort), ny = parseFloat(y.dataset.sort);
          var r = (!isNaN(nx) && !isNaN(ny))
            ? nx - ny
            : x.textContent.trim().localeCompare(y.textContent.trim());
          return asc ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    };
    th.addEventListener('click', sort);
    th.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); sort(); }
    });
  });
});
"""


#: A number with one of the report's prefixable units attached, as the measuring
#: modules write it into a note (``0.00398 N.m.s``, ``4.25e-09 N.m``). Tight on
#: purpose: it matches a value and a whole unit or nothing, so the failure mode
#: is a number left in SI rather than a corrupted sentence.
_QUANTITY = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s(N\.m\.s|N\.m|A\.m\^2)(?![\w.^])"
)


def _rescale(text: str, scales: dict[str, UnitScale]) -> str:
    """Put the quantities inside a note in the same units as the row above it.

    A row whose threshold reads ``0.2487 mN·m·s`` and whose note reads
    ``2.487e-04 N·m·s`` is asking the reader to convert, which is the thing the
    prefix was introduced to stop. The row's own scale is used where the note's
    unit is one the family carries; where it is not (a torque quoted inside a
    momentum criterion's note), the note is treated as its own family, since a
    sentence is compared against itself and against nothing in the table.

    Presentation only, and on strings this package wrote: nothing here parses a
    rendered value back into a verdict.

    Parameters
    ----------
    text : str
        The note, before any math substitution.
    scales : dict
        The row's family scales, keyed by units string.

    Returns
    -------
    str
    """
    found: dict[str, list[float]] = {}
    for value, units in _QUANTITY.findall(text):
        found.setdefault(units, []).append(float(value))
    if not found:
        return text
    chosen = {
        units: scales.get(units) or unit_scale(units, values)
        for units, values in found.items()
    }
    return _QUANTITY.sub(
        lambda m: (
            f"{chosen[m.group(2)].text(float(m.group(1)), 3)} "
            f"{chosen[m.group(2)].units}"
        ),
        text,
    )


def _criterion_note(criterion: Criterion, scales: dict[str, UnitScale]) -> str:
    """A criterion's note, scannable: the equation and its numbers, then a disclosure.

    The measuring modules write a note as "<formula> = <value> (<inputs>)" and
    then, often, a paragraph explaining the threshold. The first part is what a
    reviewer needs in the row; the paragraph is what they need only when the row
    surprises them, so it collapses.

    The equation is lifted off the front using the criterion's own
    :attr:`~analysis.common.report.Criterion.formula`, which is the same string
    the note was built from — not by matching the note against a table of
    formulae the report might contain.
    """
    text = _plain(criterion.note)
    for clause in _BOILERPLATE:
        text = text.replace(clause, "")
    text = _rescale(text, scales)
    formula = ""
    if criterion.formula and text.startswith(criterion.formula):
        formula = tex_html(criterion.formula, criterion.formula_tex)
        text = text[len(criterion.formula) :].strip()
    head, tail = _split(text)
    lead = math_html(head) if formula else _prose(head)
    return f'<div class="note">{formula} {lead}</div>' + _why(_prose(tail))


def _glossary(entries: tuple[tuple[str, str], ...]) -> str:
    """What a family's short codes mean, in the open at the head of the family.

    D1 through D4 and M1 through M3 are the report's own labels for the sizing
    drivers and the magnetorquer criteria, and a reader meeting them for the
    first time has no way to expand them from the row alone. This was a
    disclosure and stayed shut, which is the same as not being there — so it is
    now visible by default, one line per code, said once per family rather than
    repeated on every row. Each code is also expanded in the criterion label
    itself (:func:`analysis.sizing.mathfmt.criterion_label`), so the code is
    never the only identifier a row carries.

    What a code *demands* is here; what its symbols *are* is in the Nomenclature
    section, and is not repeated.
    """
    if not entries:
        return ""
    items = "".join(
        f"<dt>{math_html(term)}</dt><dd>{_prose(meaning)}</dd>"
        for term, meaning in entries
    )
    return (
        f'<dl class="glossary">{items}</dl>'
        '<p class="xref">Symbols are defined in <a href="#nomenclature">'
        "Nomenclature</a>.</p>"
    )


def _family_scales(criteria: list[Criterion]) -> dict[str, UnitScale]:
    """One SI prefix per units string across a family of criteria.

    The family, not the whole table and not the individual row: the rows a
    reader compares are the ones sitting under the same subhead, and one prefix
    over the whole table would have to serve both a wheel torque in millinewton
    metres and a secular disturbance torque five decades under it.

    Parameters
    ----------
    criteria : list of analysis.common.report.Criterion
        One family, in document order.

    Returns
    -------
    dict
        Units string to the scale every cell in that unit is set through.
    """
    families: dict[str, list[float]] = {}
    for c in criteria:
        families.setdefault(c.units, []).extend((c.threshold, c.measured, c.margin))
    return {units: unit_scale(units, values) for units, values in families.items()}


def _criterion_name(criterion: Criterion) -> str:
    """A criterion's name, its short code set apart from the words expanding it."""
    return math_html(sentence_case(criterion_label(criterion.name)))


def _criteria_table(grouped: list[list[Criterion]]) -> str:
    """The criteria table: grouped by family, sortable within each group.

    Takes the families already ordered (:func:`_document_order`) rather than the
    report, so the table and the margin chart cannot drift out of step.

    Margin is one cell carrying both the absolute figure and the percentage,
    because they answer the same question and separating them makes the reader
    do the division.

    Units are prefixed per family, not per cell (:func:`_family_scales`), so the
    threshold, the measured value and the margin of one row are always in the
    same unit and the rows of one family are directly comparable. Every one of
    the three cells states that unit: a trio that shares a unit silently is a
    trio a reader has to take on trust.
    """
    head = (
        "<thead><tr><th>Requirement</th><th>Criterion</th>"
        '<th class="n">Threshold</th><th class="n">Measured</th>'
        '<th class="n">Margin</th><th>Verdict</th></tr></thead>'
    )
    blocks = []
    for (title, why, _, glossary), criteria in zip(_FAMILIES, grouped):
        if not criteria:
            continue
        scales = _family_scales(criteria)
        failing = sum(1 for c in criteria if not c.passes)
        noun = "criterion" if len(criteria) == 1 else "criteria"
        tally = (
            f"{len(criteria)} {noun}, all pass"
            if not failing
            else f"{len(criteria)} {noun}, {failing} FAIL"
        )
        blocks.append(
            f'<tbody class="group"><tr><th colspan="6">{_esc(title)}'
            f'<span class="why">{math_html(why)} {_esc(tally)}.'
            f"{_glossary(glossary)}</span>"
            "</th></tr></tbody><tbody>"
        )
        for c in criteria:
            verdict = "PASS" if c.passes else "FAIL"
            sense = "≥" if c.sense == "min" else "≤"
            note = _criterion_note(c, scales) if c.note else ""
            scale = scales[c.units]
            units = _unit(scale.units)
            blocks.append(
                f'<tr class="{"fail" if not c.passes else ""}">'
                # No requirement ID is the normal case here and says something:
                # nothing in the baseline is written on actuator sizing. "None"
                # states that; a lower-case "n/a" reads as a missing field.
                f'<td class="req">{_esc(c.requirement) if c.requirement else "None"}</td>'
                f'<td><span class="crit">{_criterion_name(c)}</span>{note}</td>'
                f'<td class="n" data-sort="{c.threshold}">{sense} '
                f"{_esc(scale.text(c.threshold))}{units}</td>"
                f'<td class="n" data-sort="{c.measured}">'
                f"{_esc(scale.text(c.measured))}{units}</td>"
                f'<td class="n" data-sort="{c.margin}">'
                f"{_esc(signed(scale.value(c.margin)))}{units} "
                f'<span class="pct">({_esc(percent(c.margin_pct))})</span></td>'
                f'<td class="verdict {verdict.lower()}">{verdict}</td></tr>'
            )
        blocks.append("</tbody>")
    return (
        '<div class="scroll"><table class="sortable">'
        + head
        + "".join(blocks)
        + "</table></div>"
    )


def _spec_items(items: tuple[SpecItem, ...]) -> str:
    """One provenance group as a definition list, one quantity per line.

    The console renders a group as a single packed line because a terminal has
    no other option. Here each quantity gets its own row: what it is, its symbol
    typeset, and its value with the units attached in the tabular face. A
    reviewer looking for the inscribed radius should find it on a line of its
    own, not fifth in a semicolon-separated string.

    The prefix is chosen once per group per units string, so the six wheel
    quantities are comparable at a glance rather than each scaled to itself. An
    item carrying a :attr:`~analysis.sizing.report.SpecItem.note` renders it as a
    caption under the group; notes never enter a value cell.
    """
    families: dict[str, list[float]] = {}
    for item in items:
        if item.si is not None:
            families.setdefault(item.units, []).append(item.si)
    scales = {units: unit_scale(units, values) for units, values in families.items()}
    rows = []
    notes = []
    for item in items:
        if item.note:
            notes.append(f"{_prose(item.label)}: {_prose(item.note)}.")
        if item.value_tex:
            # The inertia tensor is the one quantity here that is not a scalar,
            # and it is the quantity the per-axis analysis depends on: every
            # SISO margin on this page is valid because the off-diagonal terms
            # are zero. It is shown as the matrix, zeros and all, rather than
            # asserted in prose. Its LaTeX already opens with the symbol, so the
            # symbol column would only repeat it and the row spans instead.
            rows.append(
                '<div class="q wide">'
                f'<span class="ql">{_esc(item.label)}</span>'
                f'<span class="qv">{tex_value(item.value_tex, item.value)}'
                f"{_unit(item.units)}</span></div>"
            )
            continue
        scale = scales.get(item.units)
        value, units = (
            (scale.text(item.si), scale.units)
            if scale is not None and item.si is not None
            else (item.value, item.units)
        )
        rows.append(
            '<div class="q">'
            f'<span class="ql">{_esc(item.label)}</span>'
            f'<span class="qs">{_math(item.symbol) if item.symbol else ""}</span>'
            f'<span class="qv">{math_html(value)}{_unit(units)}</span>'
            "</div>"
        )
    caption = f'<p class="qnote">{" ".join(notes)}</p>' if notes else ""
    return f'<div class="quantities">{"".join(rows)}</div>{caption}'


def _derived_cards(analysis: SizingAnalysis) -> str:
    """The derived tuning as cards — the justification, not a dump.

    The prefix is chosen per card from the pair the card exists to compare, so
    the derived and the committed value are always in the same unit; the
    comparison is the card's whole point and two prefixes would break it.
    """
    cards = []
    for p in analysis.derived:
        scale = unit_scale(p.units, [p.derived, p.committed])
        units = _unit(scale.units)
        if math.isnan(p.committed):
            committed = '<span class="v">&ndash;</span>'
            delta = '<div class="d">No committed value in the config.</div>'
        else:
            committed = f'<span class="v">{_esc(scale.text(p.committed))}</span>'
            difference = (
                float("nan") if math.isnan(p.ratio) else 100.0 * (p.ratio - 1.0)
            )
            agreement = (
                "identical"
                if abs(difference) < 0.05
                else f"{_esc(percent(difference))}"
            )
            delta = (
                f'<div class="d">Committed is <b>{_esc(_num(p.ratio, 3))}×</b> '
                f"the derived value ({agreement}).</div>"
            )
        cards.append(
            f'<div class="card"><h3>{_esc(p.name)}</h3>'
            '<div class="compare">'
            '<div class="k">Derived</div><div class="k">Committed</div>'
            f'<div><span class="v">{_esc(scale.text(p.derived))}</span>{units}</div>'
            f"<div>{committed}{units if not math.isnan(p.committed) else ''}</div>"
            f"{delta}</div>"
            f'<div class="row"><span class="k">Formula</span>'
            f'<span class="formula">{tex_html(p.formula, p.formula_tex)}</span></div>'
            f'<div class="row"><span class="k">Evaluated at</span>'
            f"{tex_html(p.inputs, p.inputs_tex)}</div>"
            f"{_lead_and_why(p.reasoning, 'Why this value')}</div>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _budget_table(analysis: SizingAnalysis) -> tuple[str, str]:
    """The disturbance budget with its formulae, beside the interactive bars.

    Returns the table and the display units its three numeric columns are in,
    so the section lede can name that unit once instead of the table repeating
    it on every cell.
    """
    budget = analysis.budget
    scale = unit_scale(
        "N.m",
        [t.torque_nm for t in budget.terms] + [budget.total_nm],
    )
    units = unit_html(scale.units)
    rows = [
        f'<thead><tr><th>Term</th><th class="n">Torque [{units}]</th>'
        f'<th class="n">Secular [{units}]</th><th class="n">Cyclic [{units}]</th>'
        "<th>Formula and inputs</th></tr></thead><tbody>"
    ]
    for t in budget.terms:
        rows.append(
            f'<tr><td><span class="crit">{_esc(sentence_case(t.name))}</span></td>'
            f'<td class="n">{_esc(scale.text(t.torque_nm))}</td>'
            f'<td class="n">{_esc(scale.text(t.secular_nm))}</td>'
            f'<td class="n">{_esc(scale.text(t.cyclic_nm))}</td>'
            f"<td>{tex_html(t.formula, t.formula_tex)}"
            f'<div class="note">{_prose(t.inputs)}</div></td></tr>'
        )
    rows.append(
        "</tbody><tfoot><tr><td>Total</td>"
        f'<td class="n">{_esc(scale.text(budget.total_nm))}</td>'
        f'<td class="n">{_esc(scale.text(budget.secular_nm))}</td>'
        f'<td class="n">{_esc(scale.text(budget.cyclic_nm))}</td>'
        '<td class="note">Summed, not RSS: worst cases can coincide.</td>'
        "</tr></tfoot>"
    )
    return '<div class="scroll"><table>' + "".join(rows) + "</table></div>", units


def _embed_png(path: Path) -> str:
    """A matplotlib figure as an inline ``data:`` URI, so the file stays one file."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_block(fig: go.Figure, caption: str, detail: str, first: bool) -> str:
    """One plotly figure, a one-line caption and its detail; plotly.js inlines once."""
    div = fig.to_html(
        include_plotlyjs="inline" if first else False,
        full_html=False,
        default_width="100%",
        config={"displaylogo": False, "responsive": True},
    )
    return (
        f"<figure>{div}<figcaption>{_prose(caption)}"
        f"{_why(_prose(detail), 'How to read it')}</figcaption></figure>"
    )


def _thesis(analysis: SizingAnalysis, report: AnalysisReport) -> str:
    """The one sentence the page exists to say, in the words its verdict allows.

    On a pass it is the claim the envelope figure below it makes visible: the
    requirement vectors sit inside the capability surfaces. On a failure it is
    the count and the worst offender by name, because "FAIL" alone sends the
    reader hunting through a table for which row moved.

    The verdict, the count and the margin all come from the report object. The
    only thing composed here is the sentence.
    """
    failures = report.failures()
    if not failures:
        # The margin convention, stated as the design carries it: the drivers
        # are compared at 1.3x, so the claim is about the comparison and not
        # about every criterion's own margin (a bound criterion sitting exactly
        # on its limit passes at 0 %, which the table shows and this does not
        # contradict).
        excess = 100.0 * (analysis.assumptions.margin - 1.0)
        return (
            "Every sizing requirement fits inside the capability envelope, "
            f"with the {_num(excess, 3)} % design margin applied to each "
            "driver it is compared against."
        )
    worst = min(failures, key=_worst_first)
    noun = "criterion does" if len(failures) == 1 else "criteria do"
    return (
        f"<b>{len(failures)} {noun} not close.</b> The worst is "
        f"{_criterion_name(worst)}, short by "
        # The magnitude, unsigned: "short by" already carries the direction, and
        # percent() writes a leading "+" that would contradict it.
        f"{_esc(percent(abs(worst.margin_pct)).lstrip('+'))}."
    )


#: Captions for the embedded matplotlib figures, by file stem. Written per
#: figure rather than derived from the filename: the two say different things,
#: and one caption covering both said the wrong thing about one of them.
_STATIC_CAPTIONS: dict[str, tuple[str, str]] = {
    "momentum_drivers": (
        "Momentum drivers against the envelope",
        "Every judged driver at its margin, against the usable envelope and the "
        "hardware radius, on a log axis. The log scale is the only way the "
        "drivers and the envelope share one plot on this class of vehicle.",
    ),
    "magnetorquer_sizing": (
        "Magnetorquer authority and the B-dot noise floor",
        "M2 on the left: the momentum the rods can remove against the tip-off "
        "momentum they must remove. M3 on the right: the committed exit "
        "threshold against the noise floor at the weakest and mean field.",
    ),
}

#: What the four numeric columns mean, which depends on the criterion's sense
#: and is the question a first-time reader of this table actually has. Open by
#: default: a reader who does not know what "measured" is measuring cannot read
#: a single row, and one who does closes it once.
_HOW_TO_READ_A_ROW = """
<details class="howto" open><summary>How to read a row</summary><div>
<p><b>Threshold</b> is the bound the design is held to and carries its direction
in the cell: <b>&#8805;</b> for a <b>minimum-sense</b> criterion, where a
capability must exceed a requirement, and <b>&#8804;</b> for a
<b>maximum-sense</b> one, where a value must stay under a ceiling.
<b>Measured</b> is what this configuration actually provides or carries.</p>
<p><b>Margin</b> is the distance between them, signed so that positive is always
the safe direction: for a minimum-sense row it is measured minus threshold, for
a maximum-sense row it is threshold minus measured. The percentage beside it is
that same distance as a fraction of the threshold, so a margin of +50 % means
the design sits half the threshold clear of it.</p>
<p>The <b>30 % convention</b> applies to the sizing comparisons only, where a
capability is required to beat a driver by that factor, and the factor is
already inside the threshold shown. It does <b>not</b> apply to the consistency
and bound criteria: those pass anywhere on the safe side of their limit, and one
of them is designed to land on it exactly, at a margin of 0 %.</p>
</div></details>
"""

#: Every symbol this page sets, what it means, its units, and the committed
#: flight parameter it is the same quantity as. That last column is the one a
#: reviewer most needs and the one no formula carries: a page can typeset
#: ``h_envelope`` perfectly and still leave the reader guessing which key in
#: ``config/spacecraft/*.yaml`` it is. Grouped in reading order — the vehicle,
#: then what it must do, then what the actuators provide, then the tuning, then
#: the environment — because an alphabetical list of forty symbols is a lookup
#: table and this is meant to be read once, top to bottom.
#:
#: ``(latex, meaning, units, flight parameter)``. An empty parameter means the
#: symbol is an intermediate of this analysis and is committed nowhere.
_NOMENCLATURE: tuple[tuple[str, str, tuple[tuple[str, str, str, str], ...]], ...] = (
    (
        "The vehicle",
        "What the config says the spacecraft is.",
        (
            (r"J", "Body-frame inertia tensor, shown in full above", "kg.m^2", ""),
            (
                r"J_{\min},\ J_{\max}",
                "Smallest and largest principal moment",
                "kg.m^2",
                "",
            ),
            (r"m", "Dry mass", "kg", ""),
            (r"W", "Actuator distribution matrix, unit spin axes as columns", "-", ""),
            (
                r"\sigma_{\min},\ \sigma_{\max}",
                "Smallest and largest singular value of W; their ratio is the "
                "array conditioning a degenerate geometry is refused on",
                "-",
                "",
            ),
        ),
    ),
    (
        "Rates and momentum",
        "What the vehicle carries and what the wheels must take from it.",
        (
            (r"\omega", "Body angular rate", "rad/s", ""),
            (
                r"\omega_{\mathrm{tipoff}}",
                "Assumed separation tip-off rate",
                "rad/s",
                "",
            ),
            (
                r"\omega_{\mathrm{exit}}",
                "Rate B-dot hands over to the wheels at",
                "rad/s",
                "DetumbleExitRadps",
            ),
            (
                r"\omega_{\mathrm{floor}}",
                "Slowest rate B-dot can resolve against magnetometer noise",
                "rad/s",
                "",
            ),
            (
                r"\omega_{\mathrm{slew}}",
                "Commanded slew rate, when the config declares one",
                "rad/s",
                "",
            ),
            (r"h", "Stored wheel momentum", "N.m.s", ""),
            (
                r"h_{\mathrm{envelope}}",
                "Certified momentum ceiling the FSW alarms on",
                "N.m.s",
                "MomentumEnvelopeNms",
            ),
            (
                r"h_{\mathrm{usable}}",
                "min(r_in, h_envelope), the momentum the vehicle may actually use",
                "N.m.s",
                "",
            ),
            (
                r"h_{\mathrm{limit}}",
                "SISO validity bound: stored momentum at which the gyroscopic term "
                "stops being negligible at crossover",
                "N.m.s",
                "",
            ),
            (
                r"h_{\mathrm{enter}},\ h_{\mathrm{exit}}",
                "Desaturation entry and exit thresholds; the gap is the hysteresis",
                "N.m.s",
                "MomentumDesatEnterNms, MomentumDesatExitNms",
            ),
            (
                r"h_{D1},\ h_{D1b},\ h_{D2},\ h_{D3},\ h_{D4}",
                "The five momentum drivers: tip-off, post-B-dot handover, cyclic "
                "storage, secular accumulation, slew agility",
                "N.m.s",
                "",
            ),
        ),
    ),
    (
        "Actuator capability",
        "The envelope geometry, and what the rods can produce.",
        (
            (
                r"r_{\mathrm{in}}",
                "Zonotope inscribed radius, the capability guaranteed in every "
                "direction and the only figure a criterion is written on",
                "N.m.s",
                "",
            ),
            (
                r"r_{\mathrm{out}}",
                "Zonotope circumscribed radius, the best direction",
                "N.m.s",
                "",
            ),
            (r"\tau", "Torque", "N.m", ""),
            (
                r"\bar\tau_{\mathrm{mtq}}",
                "Average magnetorquer torque available for desaturation",
                "N.m",
                "",
            ),
            (
                r"m_{\mathrm{in}}",
                "Dipole the rod array guarantees in every direction",
                "A.m^2",
                "",
            ),
            (
                r"\eta",
                "Magnetorquer efficiency, the fraction of dipole that does work",
                "-",
                "",
            ),
            (
                r"d_{\mathrm{duty}}",
                "Rod duty factor, the fraction of the orbit they may drive",
                "-",
                "MtqDutyFactor",
            ),
            (
                r"m_{\mathrm{res}}",
                "Residual (uncommanded) magnetic dipole of the vehicle",
                "A.m^2",
                "",
            ),
        ),
    ),
    (
        "Control tuning",
        "The gains and periods the design implies.",
        (
            (r"K_p", "Proportional attitude gain", "N.m/rad", "PidKpNmPerRad"),
            (r"K_d", "Derivative (rate) gain", "N.m/(rad/s)", "PidKdNmPerRadps"),
            (
                r"\omega_n",
                "Closed-loop natural frequency the committed gains imply",
                "rad/s",
                "",
            ),
            (r"\zeta", "Closed-loop damping ratio the committed gains imply", "-", ""),
            (r"\omega_{c}", "Loop gain crossover frequency", "rad/s", ""),
            (
                r"\Delta t",
                "Control period, the interval between attitude updates",
                "s",
                "ControlPeriodS",
            ),
            (r"k", "B-dot gain", "N.m.s", "BdotGainNms"),
            (
                r"f",
                "Detumble exit fraction: how much of the usable envelope the handover may consume",
                "-",
                "",
            ),
            (
                r"\rho_{\max}",
                "Largest gyroscopic-to-control torque ratio the SISO analysis "
                "tolerates (MAX_SISO_COUPLING_RATIO)",
                "-",
                "",
            ),
        ),
    ),
    (
        "Orbit and environment",
        "Where the vehicle flies and what pushes on it there.",
        (
            (r"a", "Semi-major axis", "km", ""),
            (r"i", "Inclination", "deg", ""),
            (r"T,\ T_{\mathrm{orbit}}", "Orbital period", "s", ""),
            (r"T_{\mathrm{desat}}", "Interval between desaturation passes", "s", ""),
            (r"\omega_o", "Orbit mean motion", "rad/s", ""),
            (r"R", "Geocentric radius", "km", ""),
            (r"\mu", "Earth gravitational parameter", "m^3/s^2", ""),
            (
                r"B,\ |B|_{\min},\ |B|_{\max},\ |B|_{\mathrm{mean}}",
                "Geomagnetic flux density, and its extremes over the orbit",
                "uT",
                "",
            ),
            (
                r"\sigma",
                "Magnetometer noise, one standard deviation per axis",
                "nT",
                "",
            ),
            (
                r"\xi",
                "Inclination of the field to the orbit plane; 90 deg is taken as the worst case",
                "deg",
                "",
            ),
            (
                r"\tau_{\mathrm{sec}},\ \tau_{\mathrm{cyc}}",
                "Secular and cyclic halves of the disturbance budget",
                "N.m",
                "",
            ),
            (r"\rho", "Atmospheric density", "kg/m^3", ""),
            (r"V", "Orbital speed", "m/s", ""),
            (r"C_d,\ C_r", "Drag and reflectivity coefficients", "-", ""),
            (r"A", "Projected area", "m^2", ""),
            (
                r"d_{cp}",
                "Offset from the centre of mass to the centre of pressure",
                "m",
                "",
            ),
            (r"\Phi/c", "Solar radiation pressure at 1 AU", "Pa", ""),
        ),
    ),
)


def _nomenclature() -> str:
    """Every symbol on the page, defined once, with its config key where it has one."""
    blocks = []
    for title, why, rows in _NOMENCLATURE:
        blocks.append(
            f'<tbody class="group"><tr><th colspan="4">{_esc(title)}'
            f'<span class="why">{_esc(why)}</span></th></tr></tbody><tbody>'
        )
        for tex, meaning, units, parameter in rows:
            blocks.append(
                f'<tr><td class="sym">{tex_symbol(tex)}</td>'
                f"<td>{_prose(meaning)}</td>"
                f'<td class="n">{unit_html(units) or "&ndash;"}</td>'
                f'<td class="req">{_esc(parameter) if parameter else "&ndash;"}</td>'
                "</tr>"
            )
        blocks.append("</tbody>")
    head = (
        "<thead><tr><th>Symbol</th><th>Meaning</th>"
        '<th class="n">Units</th><th>Flight parameter</th></tr></thead>'
    )
    return (
        '<div class="scroll"><table class="nomen">'
        + head
        + "".join(blocks)
        + "</table></div>"
    )


#: The document's sections, in order: anchor and nav label.
_SECTIONS = (
    ("verdict", "Verdict"),
    ("criteria", "Criteria"),
    ("figures", "Figures"),
    ("budget", "Disturbance budget"),
    ("derived", "Derived parameters"),
    ("nomenclature", "Nomenclature"),
    ("assumptions", "Assumptions"),
    ("warnings", "Warnings"),
)


def write_html(
    analysis: SizingAnalysis,
    report: AnalysisReport,
    out_dir: str | Path,
    static_figures: list[Path] | None = None,
) -> Path:
    """Render the whole sizing result to ``<out_dir>/index.html``.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis; supplies the figures and the derived tuning.
    report : analysis.common.report.AnalysisReport
        The structured verdict. Every PASS/FAIL word on the page is read from
        here — the page renders the verdict, it never decides it.
    out_dir : str or pathlib.Path
        Destination directory; created if absent.
    static_figures : list of pathlib.Path, optional
        PNGs to embed inline as ``data:`` URIs — the matplotlib figures with no
        interactive value. Missing files are skipped rather than raising, so a
        ``--no-plots`` run still produces a page.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    verdict = "PASS" if report.passes else "FAIL"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # The title is "<what> — <vehicle>"; the header shows them separately so the
    # vehicle stays legible when the bar is compressed. Titles without the
    # separator degrade to a single label rather than being mangled.
    what, separator, craft = report.title.partition(" — ")
    if not separator:
        what, craft = "", report.title

    grouped = _document_order(report)
    ordered = [criterion for family in grouped for criterion in family]

    # The hero. The claim is the sentence; this figure is the claim made
    # visible, so it sits directly under it and carries the inlined plotly
    # bundle rather than one of the supporting figures.
    hero_figure = _figure_block(
        momentum_envelope_figure(analysis),
        "Each driver is an arrow from the origin; every arrow ending inside the "
        "orange surface is momentum the vehicle is certified to hold.",
        "Rotate and zoom. Orange is the certified ceiling "
        "(MomentumEnvelopeNms), blue the hardware guarantee, slate the wheel "
        "zonotope the L-infinity allocator reaches. A driver arrow reaching "
        "past the orange surface is momentum this vehicle may not use, whatever "
        "the hardware can do. On this class of vehicle the drivers are orders "
        "of magnitude smaller than the envelope, so zoom in or read the margin "
        "chart under Figures.",
        True,
    )

    figures = [
        _figure_block(
            torque_envelope_figure(analysis),
            "The same construction for torque, with demand along the array's "
            "weakest direction.",
            "Demand is PidMaxTorqueNm plus the total disturbance torque.",
            False,
        ),
        _figure_block(
            disturbance_figure(analysis),
            "Closed-form worst case at a static attitude.",
            "The secular/cyclic split is an assumption, stated under "
            "Assumptions, not a measurement.",
            False,
        ),
        _figure_block(
            margin_figure(report, ordered),
            "Every criterion above, as margin against its own threshold, in the "
            "table's order.",
            "The bar carries the verdict colour and the word PASS or FAIL, so "
            "the colour is never load-bearing alone. Zero is the threshold: a "
            "bar to the left of it is a criterion that does not close.",
            False,
        ),
    ]
    for path in static_figures or []:
        if Path(path).is_file():
            title, caption = _STATIC_CAPTIONS.get(
                Path(path).stem,
                (
                    Path(path).stem.replace("_", " ").capitalize(),
                    "A static figure written beside this page.",
                ),
            )
            figures.append(
                f'<figure><img src="{_embed_png(Path(path))}" alt="{_esc(title)}">'
                f"<figcaption><b>{_esc(title)}.</b> {_prose(caption)} Static, and "
                "the rendering that prints.</figcaption></figure>"
            )

    # Built from the structured quantities rather than the report's flattened
    # provenance strings: same numbers, one per line instead of packed into one.
    provenance = "".join(
        f"<dt>{_esc(provenance_label(key))}</dt><dd>{_spec_items(items)}</dd>"
        for key, items in spec_groups(analysis)
    )
    assumptions = "".join(f"<li>{_lead_and_why(a)}</li>" for a in report.assumptions)
    warnings = "".join(
        f"<li>{_lead_and_why(w, 'Detail')}</li>" for w in report.warnings
    )
    budget_table, budget_units = _budget_table(analysis)
    thesis = _thesis(analysis, report)
    nav = "".join(
        f'<li><a href="#{anchor}">{_esc(label)}</a></li>' for anchor, label in _SECTIONS
    )
    failures = len(report.failures())

    katex_css, katex_js = katex_assets()

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Declared in the head, not only as a CSS property: a browser with
     auto-dark-mode enabled decides whether to force-darken before it
     parses the stylesheet, so the CSS declaration alone arrives too late
     and the page is re-rendered inverted. This report is a review
     artifact whose verdict colours and figures are chosen against a light
     ground; it must look the same for every reader. -->
<meta name="color-scheme" content="light">
<title>{_esc(_plain(report.title))} ({verdict})</title>
<style>{katex_css}</style>
<style>{_CSS}</style></head><body>
<header class="topbar">
  <div class="row">
    <div class="who">
      <span class="craft">{_esc(craft)}</span>
      <span class="what">{_esc(what)}</span>
    </div>
    <span class="pill {verdict.lower()}">{verdict}</span>
    <div class="meta"><code>{_esc(report.config_path)}</code><br>{_esc(stamp)}</div>
  </div>
  <nav class="sections"><ol>{nav}</ol></nav>
</header>
<main>

<section class="hero" id="verdict">
  <div class="statement {verdict.lower()}">
    <span class="pill {verdict.lower()}">{verdict}</span>
    <h1>{thesis}</h1>
    <span class="tally">{failures} failing criteria of {len(report.criteria)}</span>
  </div>
  {hero_figure}
  <div class="spec"><dl>{provenance}</dl></div>
</section>

<section id="criteria"><h2>Criteria</h2>
<p class="lede">Criteria are grouped by what they judge and led by the tightest
margin in each group. Click a column heading to sort within a family; failing
rows are shaded <b>and</b> say FAIL. Each family opens with a one-line
definition of every short code it uses, and each row's threshold, measured value
and margin are stated in one shared unit so the comparison is direct.</p>
{_HOW_TO_READ_A_ROW}
{_criteria_table(grouped)}
</section>

<section id="figures"><h2>Figures</h2>
{"".join(figures)}
</section>

<section id="budget"><h2>Disturbance-torque budget</h2>
<p class="lede">Worst case at a static attitude, analytic. Every torque column is
in {budget_units}.</p>
{budget_table}
</section>

<section id="derived"><h2>Derived flight parameters</h2>
<p class="lede">What this design implies the tuning should be, against what the
config carries. Recommendations, not criteria: nothing here moves the verdict.</p>
{_derived_cards(analysis)}
</section>

<section id="nomenclature"><h2>Nomenclature</h2>
<p class="lede">Every symbol this page sets, with its units and, where the symbol
names a quantity the vehicle commits to, the flight parameter that carries it.
An em dash in the last column means the symbol is an intermediate of this
analysis and is committed nowhere.</p>
{_nomenclature()}
</section>

<section id="assumptions"><h2>Assumptions in force</h2>
<p class="lede">A margin without its assumptions is not a result.</p>
<div class="panel"><ul class="block">{assumptions}</ul></div>
</section>

<section id="warnings"><h2>Warnings</h2>
<p class="lede">These qualify the analysis; they do not fail it.</p>
<div class="panel warn"><ul class="block warn">{warnings}</ul></div>
</section>

</main><script>{katex_js}</script><script>{_JS}</script></body></html>
"""
    target = directory / "index.html"
    target.write_text(page, encoding="utf-8")
    return target
