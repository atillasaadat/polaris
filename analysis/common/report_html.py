"""The shared page design behind every Polaris HTML analysis report.

A report is written twice. :meth:`analysis.common.report.AnalysisReport.format_text`
is the record — fixed-width, asserted on by the tests, the thing that goes in a
log. This layer is the *other* rendering, for the case the text serves badly: a
design review, where the reader wants a document rather than a dump.

What lives here and what does not
---------------------------------
Here: the design system. One stylesheet, one behaviour script, the KaTeX and
figure plumbing, the prose helpers that turn console strings into document
prose, and the page shell that assembles a head, a sticky header carrying the
verdict, a section nav and a body. None of it knows what is being reported.

Not here: what any one analysis has to *say*. The sections, their order, the
tables inside them, the sentence at the top — those belong to the package that
computed the numbers, because they are the analysis, not its typography.
:mod:`analysis.sizing.html` and :mod:`analysis.od.html` are the two consumers and
they share no content, only this shell.

The rule the whole layer is held to
------------------------------------
**Nothing here computes a verdict.** Every PASS/FAIL word on a page comes from
the :class:`~analysis.common.report.AnalysisReport` handed in. A renderer that
could decide an outcome would be a second, untested implementation of the
analysis — which is the failure this separation exists to prevent.

Self-contained by construction
------------------------------
plotly.js is inlined into the first figure (``include_plotlyjs="inline"``),
KaTeX's stylesheet, library and eight WOFF2 faces are inlined by
:func:`analysis.common.texmath.katex_assets`, and matplotlib figures are embedded
as ``data:`` URIs. No external stylesheet, no web font, no CDN: one file that
opens offline, survives being emailed, and fetches nothing at runtime. It is a
few megabytes for exactly that reason, and
``tests/analysis/test_sizing_html.py`` is what keeps the property.

Escaping
--------
Every value that reaches a page goes through :func:`html.escape` — including
before any math substitution, see :func:`analysis.common.mathfmt.math_html`.
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
import os
import re
import shutil
import subprocess
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go

from analysis.common.mathfmt import math_html, sentence_case, unit_html
from analysis.common.texmath import katex_assets


def esc(value: object) -> str:
    """Escape any value for interpolation into the page."""
    return _html.escape(str(value), quote=True)


#: Em dashes are set for prose read at length; these pages are scanned. Each one
#: becomes a comma at render time so the source strings, which the plain-text
#: report shares, are left alone. ``**emphasis**`` is likewise console markup
#: with no meaning here.
_EM_DASH = re.compile(r"\s*—\s*")

#: A sentence boundary: ``.`` followed by whitespace only, so ``0.5 N.m.s`` and
#: ``REQ-ACTL-009`` survive.
_SENTENCE = re.compile(r"(?<=\.)\s+")

#: A clause boundary, used only on a note long enough that one sentence is
#: already a paragraph. Report notes use ``;`` where a full stop would do.
_CLAUSE = re.compile(r"(?<=;)\s+")

#: Below this, a string is short enough to read whole and is not split [chars].
_LONG = 150

#: Below this, a remainder is not worth a disclosure of its own [chars].
_WORTH_HIDING = 60


def plain(text: object) -> str:
    """A source string as the page sets prose: no em dashes, no ``**``."""
    return _EM_DASH.sub(", ", str(text)).replace("**", "")


def prose(text: object) -> str:
    """Prose, de-dashed, sentence-cased and set as mathematics."""
    return math_html(sentence_case(plain(text)))


def split_lead(text: str) -> tuple[str, str]:
    """What stays in the open, and what collapses beneath it.

    The first sentence leads. A note with no full stop splits at a semicolon
    instead, but only once it is long enough that leaving it whole would be a
    paragraph in a table cell; and a remainder too short to be worth a click
    stays where it is.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple of (str, str)
        The visible lead and the collapsed remainder, the latter empty when the
        string is better left whole.
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


def why(body: str, label: str = "Why") -> str:
    """The reasoning, one click away: what shows by default is the verdict."""
    if not body.strip():
        return ""
    return f"<details><summary>{esc(label)}</summary><div>{body}</div></details>"


def lead_and_why(text: object, label: str = "Why") -> str:
    """One sentence in the open, the remainder collapsed beneath it."""
    head, tail = split_lead(plain(text))
    return f'<div class="lead">{prose(head)}</div>' + why(prose(tail), label)


def unit(units: object) -> str:
    """A units string, set as markup and styled secondary to its number.

    ``×`` is a multiplier rather than a unit and sets tight against its number
    (``10×``); everything else takes the usual space (``0.38 N·m·s``).
    """
    rendered = unit_html(units)
    if not rendered:
        return ""
    separator = "" if rendered == "×" else " "
    return f'{separator}<span class="u">{rendered}</span>'


def math(text: object) -> str:
    """A formula or an input list, set as mathematics."""
    return f'<span class="m">{math_html(text)}</span>'


def glossary(entries: Sequence[tuple[str, str]], xref: bool = True) -> str:
    """A group's short codes, defined in the open at the head of the group.

    A reader meeting a report's private code for the first time cannot expand it
    from the row alone. This was a ``<details>`` and stayed shut, which is the
    same as not being there — so it is visible by default, one line per code,
    said once per group rather than repeated on every row.

    What a code *demands* belongs here; what its symbols *are* belongs in a
    Nomenclature section, and is not repeated.

    Parameters
    ----------
    entries : sequence of (str, str)
        Term and meaning, in the order they should read.
    xref : bool, optional
        Append the pointer to a ``#nomenclature`` section. Off for a report that
        has no such section, since a dead anchor is worse than no pointer.

    Returns
    -------
    str
        Empty when there is nothing to define.
    """
    if not entries:
        return ""
    items = "".join(
        f"<dt>{math_html(term)}</dt><dd>{prose(meaning)}</dd>"
        for term, meaning in entries
    )
    pointer = (
        '<p class="xref">Symbols are defined in <a href="#nomenclature">Nomenclature</a>.</p>'
        if xref
        else ""
    )
    return f'<dl class="glossary">{items}</dl>{pointer}'


def embed_png(path: Path) -> str:
    """A matplotlib figure as an inline ``data:`` URI, so the file stays one file."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def figure_block(fig: go.Figure, caption: str, detail: str, first: bool) -> str:
    """One plotly figure, a one-line caption and its detail; plotly.js inlines once.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    caption : str
        One sentence, shown in the open.
    detail : str
        How to read it; collapsed beneath the caption.
    first : bool
        True for exactly one figure per page — the one that carries the inlined
        plotly bundle. Passing it more than once inlines megabytes twice.

    Returns
    -------
    str
    """
    div = fig.to_html(
        include_plotlyjs="inline" if first else False,
        full_html=False,
        default_width="100%",
        config={"displaylogo": False, "responsive": True},
    )
    return (
        f"<figure>{div}<figcaption>{prose(caption)}"
        f"{why(prose(detail), 'How to read it')}</figcaption></figure>"
    )


def static_figure(path: Path, title: str, caption: str) -> str:
    """A PNG written beside the page, embedded inline with its caption."""
    return (
        f'<figure><img src="{embed_png(path)}" alt="{esc(title)}">'
        f"<figcaption><b>{esc(title)}.</b> {prose(caption)} Static, and "
        "the rendering that prints.</figcaption></figure>"
    )


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

CSS = """
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
/* A record identifier set beside its human title: a scenario or field name a
   reader may need to grep the records for, styled as data rather than prose. */
.code {
  font-family: var(--mono); font-size: .72em; font-weight: 400;
  color: var(--muted); letter-spacing: 0;
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


JS = """
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


def render_page(
    *,
    title: str,
    verdict: str,
    config_path: object,
    sections: Sequence[tuple[str, str]],
    body: str,
    stamp: str | None = None,
) -> str:
    """Assemble a complete, self-contained report page.

    The caller owns the body — every ``<section>`` in it, in the order it wants
    them — because the sections *are* the analysis. What this owns is everything
    around them: the head, the stylesheet, the KaTeX and behaviour scripts, and
    the sticky header carrying the vehicle, the verdict and the provenance.

    Parameters
    ----------
    title : str
        The report title, conventionally ``"<what> — <vehicle>"``. The header
        sets the two apart so the vehicle stays legible when the bar is
        compressed; a title without the separator degrades to a single label
        rather than being mangled.
    verdict : str
        ``"PASS"`` or ``"FAIL"``. Read from the report, never decided here.
    config_path : object
        Shown in the header as the provenance. Escaped like everything else.
    sections : sequence of (str, str)
        Anchor and label for the nav, in document order. The anchors must exist
        in ``body``; nothing here checks that, because a missing one is visible
        the first time anyone clicks it.
    body : str
        The complete contents of ``<main>``.
    stamp : str, optional
        Render time; defaults to now, in UTC. Passed explicitly by tests that
        need a byte-stable page.

    Returns
    -------
    str
        The whole document, ready to write.
    """
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    what, separator, craft = title.partition(" — ")
    if not separator:
        what, craft = "", title
    nav = "".join(
        f'<li><a href="#{anchor}">{esc(label)}</a></li>' for anchor, label in sections
    )
    katex_css, katex_js = katex_assets()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Declared in the head, not only as a CSS property: a browser with
     auto-dark-mode enabled decides whether to force-darken before it
     parses the stylesheet, so the CSS declaration alone arrives too late
     and the page is re-rendered inverted. This report is a review
     artifact whose verdict colours and figures are chosen against a light
     ground; it must look the same for every reader. -->
<meta name="color-scheme" content="light">
<title>{esc(plain(title))} ({verdict})</title>
<style>{katex_css}</style>
<style>{CSS}</style></head><body>
<header class="topbar">
  <div class="row">
    <div class="who">
      <span class="craft">{esc(craft)}</span>
      <span class="what">{esc(what)}</span>
    </div>
    <span class="pill {verdict.lower()}">{verdict}</span>
    <div class="meta"><code>{esc(config_path)}</code><br>{esc(stamp)}</div>
  </div>
  <nav class="sections"><ol>{nav}</ol></nav>
</header>
<main>
{body}
</main><script>{katex_js}</script><script>{JS}</script></body></html>
"""


def _is_wsl() -> bool:
    """True on Windows Subsystem for Linux.

    WSL reports a Microsoft kernel in ``/proc/version``; the environment
    variable is only set for interactive shells, so the kernel string is the
    reliable test.
    """
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def open_in_browser(page: Path, *, enabled: bool = True) -> None:
    """Open *page*, or say plainly how to open it, without ever failing the run.

    Two modes, and the default differs by caller on purpose. A person running
    the CLI wants the page; a test, a CI job or an agent regenerating the report
    twenty times wants the file and nothing else — a browser window opening
    unbidden is at best noise and at worst a stalled pipeline waiting on a
    process that never exits.

    So opening is **opt-out per call** (``--no-browser`` on every report CLI) and
    **opt-out globally** via ``POLARIS_NO_BROWSER``: set that in an environment
    and nothing here will ever open a window, whatever any caller asks for. The
    path is always printed either way, so a save-only run still tells the caller
    where the artifact is.

    :mod:`webbrowser` assumes a browser inside the machine it runs on. A WSL
    distro usually has none: the browser is on the Windows side, so the module
    falls through to ``gio``/``xdg-open`` and those report "no application for
    text/html". The report has already been written at that point, so a failure
    to *display* it must not look like a failure to *produce* it — the analysis
    exit code belongs to the criteria, not to the desktop.

    On WSL the page is handed to the Windows shell (``wslview`` if the wslu
    package is installed, otherwise ``explorer.exe`` on the translated path),
    which opens the user's real browser. Everywhere else :mod:`webbrowser` is
    correct and is used unchanged.

    Parameters
    ----------
    page : pathlib.Path
        The written report.
    enabled : bool, optional
        False to save only. ``POLARIS_NO_BROWSER`` overrides True back to False;
        nothing overrides False back to True.
    """
    target = str(page.resolve())
    if not enabled or os.environ.get("POLARIS_NO_BROWSER", "").strip():
        return
    if not _is_wsl():
        if not webbrowser.open(page.resolve().as_uri()):
            print(f"could not open a browser; the report is at {target}")
        return

    if shutil.which("wslview"):
        if subprocess.run(["wslview", target], check=False).returncode == 0:
            return
    windows_path = ""
    if shutil.which("wslpath"):
        translated = subprocess.run(
            ["wslpath", "-w", target], capture_output=True, text=True, check=False
        )
        if translated.returncode == 0:
            windows_path = translated.stdout.strip()
    if windows_path and shutil.which("explorer.exe"):
        # explorer.exe exits 1 even on success, so its status says nothing;
        # what matters is whether the call could be made at all.
        try:
            subprocess.run(["explorer.exe", windows_path], check=False)
            return
        except OSError:
            pass
    print(
        f"no browser reachable from WSL — open this from Windows:\n  {windows_path or target}"
    )
