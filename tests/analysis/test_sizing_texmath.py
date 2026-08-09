"""The LaTeX typesetting layer of the HTML sizing report (design doc §12).

Two properties, and they are the two that decay silently. **Coverage**: every
formula string the analysis emits has a LaTeX form, so a formula edited upstream
is caught here rather than quietly degrading to the Unicode fallback on the page.
**Renderability**: every LaTeX form actually sets, because ``mathtext`` is a
subset of LaTeX and a construct it does not implement fails at render time, not
at import.
"""

from __future__ import annotations

import pytest

from analysis.sizing.mathfmt import math_html
from analysis.sizing.report import sizing_analysis
from analysis.sizing.texmath import TEX, split_leading_formula, tex_html


@pytest.fixture(scope="module")
def analysis(vehicle):
    """The reference vehicle's sizing analysis, computed once for this module."""
    return sizing_analysis(vehicle)


def test_every_formula_the_analysis_emits_has_a_latex_form(analysis):
    """The lookup table covers the report, or the page falls back without saying so."""
    emitted = (
        [p.formula for p in analysis.derived]
        + [t.formula for t in analysis.budget.terms]
        + [d.formula for d in analysis.wheels.drivers]
    )
    missing = sorted({f for f in emitted if f not in TEX})
    assert not missing, f"no LaTeX for: {missing}"


@pytest.mark.parametrize("formula", sorted(TEX))
def test_every_latex_form_sets(formula):
    """mathtext is a LaTeX subset: a construct it lacks fails here, not on the page.

    The rendered element must also carry the geometry that puts it on the
    baseline of the line it sits in, since an equation floating half a line
    high is worse than the Unicode it replaced.
    """
    rendered = tex_html(formula)
    assert rendered.startswith('<img class="tex" src="data:image/svg+xml;base64,')
    assert "height:" in rendered and "vertical-align:" in rendered
    assert 'alt="' in rendered


def test_an_unknown_formula_falls_back_instead_of_failing():
    """A formula with no LaTeX form still renders; the report never crashes on prose."""
    assert tex_html("some new formula") == math_html("some new formula")
    assert tex_html("N.m.s per wibble") == math_html("N.m.s per wibble")


def test_a_note_opening_with_a_formula_splits_into_equation_and_numbers():
    """Criterion notes are "<formula> = <value> (<inputs>)"; the row sets both parts."""
    formula, rest = split_leading_formula(
        "tau_secular * T_desat = 2.41e-05 N.m.s (tau_secular = 4.25e-09 N.m)"
    )
    assert formula.startswith('<img class="tex"')
    assert rest == "= 2.41e-05 N.m.s (tau_secular = 4.25e-09 N.m)"


def test_a_note_that_is_prose_is_left_whole():
    """No formula, no split: the fallback is the note unchanged, not a guess at one."""
    note = "the wheels themselves, ignoring the flight envelope"
    assert split_leading_formula(note) == ("", note)
