"""The interactive HTML rendering of the sizing report (design doc §12).

The page is a *rendering*, so nothing here reads a verdict out of it. Every
pass/fail assertion is made on the structured
:class:`~analysis.common.report.AnalysisReport`, per ``analysis/CLAUDE.md``, and
what the HTML is tested for is the three properties the markup itself must
carry: that it is self-contained, that it says the same words the report object
says, and that it escapes what a config can put into it.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

from analysis.control.vehicle import load_vehicle
from analysis.sizing.html import _plain, _prose, _split, write_html
from analysis.sizing.mathfmt import math_html, sentence_case
from analysis.sizing.plots import driver_figure
from analysis.sizing.report import sizing_analysis, sizing_report
from analysis.sizing.texmath import tex_html


def _both_halves(text: str) -> tuple[str, ...]:
    """The lead sentence and the collapsed remainder, as the page sets them.

    The page shows one sentence and folds the rest into a ``<details>``. Both
    halves must still be *on* the page: collapsing the justification is a
    layout decision, deleting it would be a content one.
    """
    head, tail = _split(_plain(text))
    return tuple(_prose(part) for part in (head, tail) if part)


@pytest.fixture(scope="module")
def analysis(vehicle):
    """The reference vehicle's sizing analysis, computed once for this module."""
    return sizing_analysis(vehicle)


@pytest.fixture(scope="module")
def report(vehicle, reference_config, analysis):
    """The structured verdict — the only thing verdicts are asserted on."""
    return sizing_report(vehicle, reference_config, analysis.assumptions, analysis)


@pytest.fixture(scope="module")
def page(analysis, report, tmp_path_factory) -> str:
    """The rendered page, written once with a real static figure, read back."""
    directory = tmp_path_factory.mktemp("html")
    png = driver_figure(analysis, directory)
    target = write_html(analysis, report, directory, static_figures=[png])
    assert target.name == "index.html"
    assert target.is_file()
    return target.read_text(encoding="utf-8")


def test_the_page_fetches_nothing_at_runtime(page):
    """One file, offline: no CDN script, stylesheet, font or image.

    An analysis artifact that only renders while its author's network is up is
    not an artifact; this one has to survive being emailed and opened in six
    months. plotly.js is inlined and the matplotlib figures are ``data:`` URIs,
    so the only acceptable number of remote references is zero.
    """
    # What must not exist is an *asset* the browser goes and gets: a script,
    # stylesheet, font or image on a remote host. Ordinary <a href> hyperlinks
    # are not fetches, and plotly's bundle carries a few of its own.
    assets = re.findall(
        r"<(?:script|link|img|iframe|source)\b[^>]*?"
        r"(?:src|href)\s*=\s*[\"\']https?://[^\"\']+",
        page,
        flags=re.IGNORECASE,
    )
    assert assets == []
    # The bundle itself must be present, or every figure is a blank frame.
    assert "Plotly" in page
    assert "data:image/png;base64," in page


def test_every_criterion_appears_with_the_verdict_the_report_gives_it(report, page):
    """The words on the page are the report object's words, criterion by criterion.

    "The report object's words" allows for the presentation layer setting them
    for a document rather than a terminal — sentence case, ``N·m·s`` for
    ``N.m.s``. What it does not allow is a criterion going missing, so the
    comparison is against :func:`~analysis.sizing.mathfmt.math_html` of the
    name, which is the identity on anything the formatter does not recognise.
    """
    for criterion in report.criteria:
        assert math_html(sentence_case(criterion.name)) in page
    # The overall verdict banner, and one FAIL/PASS cell per criterion. The
    # verdict counts come from the structured report, never from the markup.
    assert page.count(">PASS<") >= sum(1 for c in report.criteria if c.passes)
    assert page.count(">FAIL<") >= sum(1 for c in report.criteria if not c.passes)
    assert f"{len(report.failures())} failing criteria" in page


def test_a_failing_design_is_labelled_in_words_not_only_in_colour(
    vehicle, reference_config, tmp_path
):
    """A design that cannot hold its tip-off renders FAIL, and the banner says so.

    The convention is that colour is never load-bearing alone. This is the
    failing side of the page: a criterion nobody has seen fail is a criterion
    nobody has tested.
    """
    weak = dataclasses.replace(vehicle, momentum_envelope_nms=1.0e-6)
    weak_analysis = sizing_analysis(weak)
    weak_report = sizing_report(weak, reference_config, analysis=weak_analysis)
    assert not weak_report.passes  # asserted on the object, not the markup

    text = write_html(weak_analysis, weak_report, tmp_path).read_text(encoding="utf-8")
    assert 'class="statement fail"' in text
    assert 'class="pill fail"' in text
    assert f"{len(weak_report.failures())} failing criteria" in text
    # The individual rows, not only the headline: a shaded row is not a verdict.
    assert text.count(">FAIL<") >= len(weak_report.failures())
    # The thesis names the count and the worst criterion, so a reader knows
    # what failed without opening the table.
    worst = min(weak_report.failures(), key=lambda c: c.margin_pct)
    assert f"{len(weak_report.failures())} criteria do not close" in text
    assert math_html(sentence_case(worst.name)) in text


def test_a_config_name_with_html_metacharacters_is_escaped(vehicle, analysis, tmp_path):
    """Config-derived strings are untrusted input and never reach the page live.

    A spacecraft name is whatever someone typed into a YAML file. If it lands in
    the markup unescaped, the report can be made to say something its numbers do
    not.
    """
    hostile = '<script>alert("x")</script>'
    named = dataclasses.replace(vehicle, name=hostile)
    hostile_report = sizing_report(named, f"config/{hostile}.yaml", analysis=analysis)

    text = write_html(analysis, hostile_report, tmp_path).read_text(encoding="utf-8")
    assert "<script>alert" not in text
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in text


def test_the_assumptions_and_the_warnings_are_both_on_the_page(report, page):
    """A margin without its assumptions is not a result (``analysis/CLAUDE.md``)."""
    assert report.assumptions and report.warnings
    for line in list(report.assumptions) + list(report.warnings):
        for half in _both_halves(line):
            assert half in page


def test_every_derived_parameter_carries_its_justification(analysis, page):
    """The cards are the "why", so the formula and the reasoning must both render."""
    assert analysis.derived
    for parameter in analysis.derived:
        assert parameter.name in page
        assert tex_html(parameter.formula) in page
        for half in _both_halves(parameter.reasoning):
            assert half in page


def test_the_page_carries_no_em_dash(page):
    """Em dashes are console prose; a page that is scanned reads without them.

    The source strings keep theirs, because the plain-text report is a record
    and rewriting it to suit a stylesheet would be the tail wagging the dog.
    The substitution is a render-time one, so this is the assertion that it
    actually reached every string on the page.
    """
    assert "—" not in page
    assert "&mdash;" not in page


def test_every_formula_is_typeset_rather_than_approximated(analysis, page):
    """Formulae render as LaTeX, and the page still fetches nothing to do it.

    ``tests/analysis/test_sizing_texmath.py`` owns the coverage of the LaTeX
    table; what matters here is that the page uses it, and that using it did
    not smuggle in a remote asset (the fetch test above would catch a CDN, this
    catches an SVG that was never rendered at all).
    """
    for parameter in analysis.derived:
        assert tex_html(parameter.formula).startswith('<img class="tex"')
    for term in analysis.budget.terms:
        assert tex_html(term.formula) in page
    assert page.count('<img class="tex"') >= len(analysis.derived)


def test_no_browser_writes_the_page_without_opening_one(
    reference_config, tmp_path, monkeypatch
):
    """``--no-browser`` is what CI and this suite use; it must not call out.

    The default *does* open a browser, which is the point of the feature — so
    the flag that suppresses it is the one worth a test, because a CI job that
    silently spawns a browser is a CI job that hangs.
    """
    from analysis.sizing import __main__ as cli

    # The exit status is the gate's, not this test's business: the reference
    # vehicle fails three criteria as committed, so the expectation is read from
    # the structured report rather than hard-coded.
    gate = sizing_report(load_vehicle(reference_config), reference_config)
    expected = 0 if gate.passes else 1

    # The seam is the opener, not `webbrowser.open`: which mechanism actually
    # reaches a browser is platform-dependent (a WSL distro has none of its own
    # and hands the page to the Windows shell instead), and a test that pins the
    # mechanism fails on the platform the feature was fixed for.
    opened: list[Path] = []
    monkeypatch.setattr(cli, "_open_in_browser", opened.append)
    monkeypatch.setattr(sys, "argv", ["analysis.sizing"])

    status = cli.main(
        [str(reference_config), "--out", str(tmp_path), "--no-plots", "--no-browser"]
    )
    assert status == expected
    assert opened == []
    assert (tmp_path / "index.html").is_file()

    status = cli.main([str(reference_config), "--out", str(tmp_path), "--no-plots"])
    assert status == expected
    assert len(opened) == 1
    assert opened[0].name == "index.html"
