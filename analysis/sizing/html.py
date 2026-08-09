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

Scannable by default, complete on demand
----------------------------------------
A reviewer reads a criteria table to find the row that surprises them, not to
read prose. So each row, card, caption and warning shows **one sentence** plus
its numbers, and everything longer collapses into a ``<details>`` disclosure.
Nothing is deleted: the justifications are the reason this report exists, they
are simply one click away rather than in the way of the next row.

The console strings are set for a fixed-width terminal, so they are typeset on
the way in. Formulae go through :mod:`analysis.sizing.texmath`, which renders
real LaTeX to an inline SVG with matplotlib mathtext (no MathJax, no KaTeX, no
web font); everything else goes through :mod:`analysis.sizing.mathfmt`, which
sets ``N.m.s`` as ``N·m·s`` and is also the fallback when a formula has no LaTeX
form. Em dashes and ``**emphasis**`` are console conventions and are dropped
here. All of it is presentation-only and lives entirely in this layer: the
report objects and the plain-text rendering keep their original strings.

Self-contained by construction
------------------------------
plotly.js is inlined into the first figure (``include_plotlyjs="inline"``) and
the matplotlib figures are embedded as ``data:`` URIs. No external stylesheet,
no web font, no CDN. The result is one file that opens offline, survives being
emailed, and fetches nothing at runtime. It is a few megabytes for exactly that
reason.

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
    _num,
    disturbance_figure,
    margin_figure,
    momentum_envelope_figure,
    torque_envelope_figure,
)
from analysis.sizing.mathfmt import (
    math_html,
    percent,
    provenance_label,
    sentence_case,
    signed,
    unit_html,
)
from analysis.sizing.report import SizingAnalysis
from analysis.sizing.texmath import split_leading_formula, tex_html


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
_FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Wheel momentum",
        "Can the array hold what the sizing drivers accumulate? "
        "Usable capability is min(zonotope r_in, MomentumEnvelopeNms).",
        ("D1", "D2", "D3", "oversizing", "handover"),
    ),
    (
        "Wheel torque",
        "Is there control torque in every direction, disturbances included?",
        ("wheel torque", "commanded torque"),
    ),
    (
        "Magnetorquer authority",
        "Can the rods desaturate the wheels and detumble the vehicle?",
        ("M1 ", "M2 ", "M3 "),
    ),
    (
        "Control tuning and thresholds",
        "Are the committed flight parameters self-consistent and in bounds?",
        (),
    ),
)


def _family_of(criterion: Criterion) -> int:
    """Index into :data:`_FAMILIES`; the last family is the catch-all."""
    for index, (_, _, keywords) in enumerate(_FAMILIES):
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

/* ---- mathematics ---- */
.m { font-family: var(--body); font-style: italic; letter-spacing: .01em; }
.m sub, .m sup { font-style: normal; font-size: .68em; }
.formula {
  display: block; background: var(--slate-soft); border-left: 2px solid var(--slate);
  padding: .4rem .65rem; margin: 0; font-size: 1.02rem; color: var(--ink);
  overflow-x: auto; white-space: nowrap;
}
/* Typeset equations, rendered server-side to inline SVG. Height and baseline
   offset are set per image in em, so an equation shrinks with the text it sits
   in and its baseline lands on the line's. */
img.tex { max-width: 100%; }

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


def _criterion_note(note: str) -> str:
    """A criterion's note, scannable: the equation and its numbers, then a disclosure.

    The measuring modules write a note as "<formula> = <value> (<inputs>)" and
    then, often, a paragraph explaining the threshold. The first part is what a
    reviewer needs in the row; the paragraph is what they need only when the row
    surprises them, so it collapses.
    """
    text = _plain(note)
    for clause in _BOILERPLATE:
        text = text.replace(clause, "")
    formula, remainder = split_leading_formula(text)
    head, tail = _split(remainder)
    lead = math_html(head) if formula else _prose(head)
    return f'<div class="note">{formula} {lead}</div>' + _why(_prose(tail))


def _criteria_table(grouped: list[list[Criterion]]) -> str:
    """The criteria table: grouped by family, sortable within each group.

    Takes the families already ordered (:func:`_document_order`) rather than the
    report, so the table and the margin chart cannot drift out of step.

    Margin is one cell carrying both the absolute figure and the percentage,
    because they answer the same question and separating them makes the reader
    do the division.
    """
    head = (
        "<thead><tr><th>Requirement</th><th>Criterion</th>"
        '<th class="n">Threshold</th><th class="n">Measured</th>'
        '<th class="n">Margin</th><th>Verdict</th></tr></thead>'
    )
    blocks = []
    for (title, why, _), criteria in zip(_FAMILIES, grouped):
        if not criteria:
            continue
        failing = sum(1 for c in criteria if not c.passes)
        noun = "criterion" if len(criteria) == 1 else "criteria"
        tally = (
            f"{len(criteria)} {noun}, all pass"
            if not failing
            else f"{len(criteria)} {noun}, {failing} FAIL"
        )
        blocks.append(
            f'<tbody class="group"><tr><th colspan="6">{_esc(title)}'
            f'<span class="why">{math_html(why)} {_esc(tally)}.</span>'
            "</th></tr></tbody><tbody>"
        )
        for c in criteria:
            verdict = "PASS" if c.passes else "FAIL"
            sense = "≥" if c.sense == "min" else "≤"
            note = _criterion_note(c.note) if c.note else ""
            blocks.append(
                f'<tr class="{"fail" if not c.passes else ""}">'
                f'<td class="req">{_esc(c.requirement) if c.requirement else "n/a"}</td>'
                f'<td><span class="crit">{math_html(sentence_case(c.name))}</span>{note}</td>'
                f'<td class="n" data-sort="{c.threshold}">{sense} {_esc(_num(c.threshold))}'
                f"{_unit(c.units)}</td>"
                f'<td class="n" data-sort="{c.measured}">{_esc(_num(c.measured))}</td>'
                f'<td class="n" data-sort="{c.margin}">{_esc(signed(c.margin))}'
                f'{_unit(c.units)} <span class="pct">({_esc(percent(c.margin_pct))})</span></td>'
                f'<td class="verdict {verdict.lower()}">{verdict}</td></tr>'
            )
        blocks.append("</tbody>")
    return (
        '<div class="scroll"><table class="sortable">'
        + head
        + "".join(blocks)
        + "</table></div>"
    )


def _derived_cards(analysis: SizingAnalysis) -> str:
    """The derived tuning as cards — the justification, not a dump."""
    cards = []
    for p in analysis.derived:
        if math.isnan(p.committed):
            committed = '<span class="v">&mdash;</span>'
            delta = '<div class="d">No committed value in the config.</div>'
        else:
            committed = f'<span class="v">{_esc(_num(p.committed))}</span>'
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
            f'<div><span class="v">{_esc(_num(p.derived))}</span>{_unit(p.units)}</div>'
            f"<div>{committed}{_unit(p.units) if not math.isnan(p.committed) else ''}</div>"
            f"{delta}</div>"
            f'<div class="row"><span class="k">Formula</span>'
            f'<span class="formula">{tex_html(p.formula)}</span></div>'
            f'<div class="row"><span class="k">Evaluated at</span>{_math(p.inputs)}</div>'
            f"{_lead_and_why(p.reasoning, 'Why this value')}</div>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _budget_table(analysis: SizingAnalysis) -> str:
    """The disturbance budget with its formulae, beside the interactive bars."""
    rows = [
        '<thead><tr><th>Term</th><th class="n">Torque</th><th class="n">Secular</th>'
        '<th class="n">Cyclic</th><th>Formula and inputs</th></tr></thead><tbody>'
    ]
    for t in analysis.budget.terms:
        rows.append(
            f'<tr><td><span class="crit">{_esc(sentence_case(t.name))}</span></td>'
            f'<td class="n">{_esc(_num(t.torque_nm * 1e6))}{_unit("uN.m")}</td>'
            f'<td class="n">{_esc(_num(t.secular_nm * 1e6))}</td>'
            f'<td class="n">{_esc(_num(t.cyclic_nm * 1e6))}</td>'
            f"<td>{tex_html(t.formula)}"
            f'<div class="note">{_prose(t.inputs)}</div></td></tr>'
        )
    budget = analysis.budget
    rows.append(
        "</tbody><tfoot><tr><td>Total</td>"
        f'<td class="n">{_esc(_num(budget.total_nm * 1e6))}{_unit("uN.m")}</td>'
        f'<td class="n">{_esc(_num(budget.secular_nm * 1e6))}</td>'
        f'<td class="n">{_esc(_num(budget.cyclic_nm * 1e6))}</td>'
        '<td class="note">Summed, not RSS: worst cases can coincide.</td>'
        "</tr></tfoot>"
    )
    return '<div class="scroll"><table>' + "".join(rows) + "</table></div>"


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
            f"judged with the {_num(excess, 3)} % design margin applied."
        )
    worst = min(failures, key=_worst_first)
    noun = "criterion does" if len(failures) == 1 else "criteria do"
    return (
        f"<b>{len(failures)} {noun} not close.</b> The worst is "
        f"{math_html(sentence_case(worst.name))}, short by "
        f"{_esc(percent(abs(worst.margin_pct)))}."
    )


#: The document's sections, in order: anchor and nav label.
_SECTIONS = (
    ("verdict", "Verdict"),
    ("criteria", "Criteria"),
    ("figures", "Figures"),
    ("budget", "Disturbance budget"),
    ("derived", "Derived parameters"),
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
            label = Path(path).stem.replace("_", " ")
            figures.append(
                f'<figure><img src="{_embed_png(Path(path))}" alt="{_esc(label)}">'
                f"<figcaption>{_esc(sentence_case(label))}: the same verdicts on "
                "a log axis, which is the only way the drivers and the envelope "
                "share one plot. Static, and the version that prints."
                "</figcaption></figure>"
            )

    provenance = "".join(
        f"<dt>{_esc(provenance_label(k))}</dt><dd>{math_html(v)}</dd>"
        for k, v in report.provenance.items()
    )
    assumptions = "".join(f"<li>{_lead_and_why(a)}</li>" for a in report.assumptions)
    warnings = "".join(
        f"<li>{_lead_and_why(w, 'Detail')}</li>" for w in report.warnings
    )
    thesis = _thesis(analysis, report)
    nav = "".join(
        f'<li><a href="#{anchor}">{_esc(label)}</a></li>' for anchor, label in _SECTIONS
    )
    failures = len(report.failures())

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(_plain(report.title))} ({verdict})</title>
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
<p class="lede">Margins are signed: positive is how far past its threshold the
design sits. Criteria are grouped by what they judge and led by the tightest
margin in each group. Click a column heading to sort within a family; failing
rows are shaded <b>and</b> say FAIL.</p>
{_criteria_table(grouped)}
</section>

<section id="figures"><h2>Figures</h2>
{"".join(figures)}
</section>

<section id="budget"><h2>Disturbance-torque budget</h2>
<p class="lede">Worst case at a static attitude, analytic. Torques in µN·m.</p>
{_budget_table(analysis)}
</section>

<section id="derived"><h2>Derived flight parameters</h2>
<p class="lede">What this design implies the tuning should be, against what the
config carries. Recommendations, not criteria: nothing here moves the verdict.</p>
{_derived_cards(analysis)}
</section>

<section id="assumptions"><h2>Assumptions in force</h2>
<p class="lede">A margin without its assumptions is not a result.</p>
<div class="panel"><ul class="block">{assumptions}</ul></div>
</section>

<section id="warnings"><h2>Warnings</h2>
<p class="lede">These qualify the analysis; they do not fail it.</p>
<div class="panel warn"><ul class="block warn">{warnings}</ul></div>
</section>

</main><script>{_JS}</script></body></html>
"""
    target = directory / "index.html"
    target.write_text(page, encoding="utf-8")
    return target
