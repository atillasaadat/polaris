"""The LaTeX typesetting layer of the HTML sizing report (design doc §12).

Three properties, and they are the three that decay silently.

**Coverage** — every object that renders a formula carries the LaTeX for it at
its own source. The previous design keyed a lookup table off the ASCII formula
strings, which rotted every time a formula was reworded: the table still matched
nothing, the page still rendered, and the reader got raw words where mathematics
was promised. Carrying the LaTeX on the object makes the two impossible to
separate, and this module asserts it stays that way.

**Renderability** — every LaTeX form actually sets. KaTeX implements a subset of
LaTeX, and a construct it lacks fails at view time, in a browser, silently
falling back to the ASCII. The vendored ``katex.min.js`` is a Node module as well
as a browser one, so the same library that renders the page validates the strings
here. Skipped where Node is unavailable rather than made a hard dependency: this
repository has no JavaScript toolchain and is not acquiring one.

**Degradation** — a formula with *no* LaTeX renders as plain styled text and is
marked as such, never as pseudo-mathematics assembled from a guess.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from analysis.sizing.html import _NOMENCLATURE
from analysis.sizing.mathfmt import math_html
from analysis.sizing.report import sizing_analysis, sizing_report, spec_groups
from analysis.sizing.texmath import _VENDOR, tex_html, tex_symbol, tex_value

#: Renders every string through the vendored KaTeX exactly as the page does,
#: except with ``throwOnError`` on: the page must degrade, a test must not.
_VALIDATE_JS = """
const katex = require(process.argv[1]);
const items = JSON.parse(process.argv[2]);
const failures = [];
for (const tex of items) {
  try { katex.renderToString(tex, { throwOnError: true, strict: 'error' }); }
  catch (error) { failures.push(tex + ' -- ' + error.message); }
}
process.stdout.write(JSON.stringify(failures));
"""


@pytest.fixture(scope="module")
def analysis(vehicle):
    """The reference vehicle's sizing analysis, computed once for this module."""
    return sizing_analysis(vehicle)


@pytest.fixture(scope="module")
def report(vehicle, analysis, reference_config):
    """The structured report, for the criteria and the formulae they carry."""
    return sizing_report(vehicle, reference_config, analysis=analysis)


def _every_tex(analysis, report) -> list[str]:
    """Every LaTeX string the page can set, from every source that carries one."""
    found = (
        [p.formula_tex for p in analysis.derived]
        + [p.inputs_tex for p in analysis.derived]
        + [t.formula_tex for t in analysis.budget.terms]
        + [d.formula_tex for d in analysis.wheels.drivers]
        + [c.formula_tex for c in report.criteria]
        + [item.value_tex for _, items in spec_groups(analysis) for item in items]
        + [tex for _, _, rows in _NOMENCLATURE for tex, _, _, _ in rows]
    )
    return sorted({tex for tex in found if tex})


def test_every_derived_parameter_carries_the_latex_for_its_formula(analysis):
    """A derived parameter is an argument; its formula is the argument's first line."""
    missing = sorted(p.name for p in analysis.derived if not p.formula_tex)
    assert not missing, f"no formula_tex on derived parameters: {missing}"


def test_every_disturbance_term_carries_the_latex_for_its_formula(analysis):
    """Each budget row shows a closed form, so each must have one to show."""
    missing = sorted(t.name for t in analysis.budget.terms if not t.formula_tex)
    assert not missing, f"no formula_tex on disturbance terms: {missing}"


def test_every_momentum_driver_carries_the_latex_for_its_formula(analysis):
    """Including the unjudged ones: they are reported, so they are rendered."""
    missing = sorted(d.name for d in analysis.wheels.drivers if not d.formula_tex)
    assert not missing, f"no formula_tex on momentum drivers: {missing}"


def test_a_criterion_that_declares_a_formula_also_carries_its_latex(report):
    """The page lifts the equation out of the note; it must have one to typeset.

    A criterion with no ``formula`` is prose and is rendered as prose — that is
    a legitimate state and not what this guards. What it guards is the half-done
    one: an equation declared structurally and then set as raw words.
    """
    missing = sorted(c.name for c in report.criteria if c.formula and not c.formula_tex)
    assert not missing, f"formula without formula_tex: {missing}"


def test_a_criterion_formula_is_the_prefix_of_its_own_note(report):
    """The page strips it off the front, so a formula that is not there is a bug.

    This is the check that keeps the two strings honest: the note is built by
    interpolating the formula, and nothing downstream re-derives that.
    """
    wrong = sorted(
        c.name
        for c in report.criteria
        if c.formula and not c.note.startswith(c.formula)
    )
    assert not wrong, f"formula is not the note's prefix: {wrong}"


def test_the_inertia_tensor_is_carried_as_a_matrix(analysis):
    """The page shows the 3x3 the per-axis analysis depends on, not a claim about it."""
    items = dict(spec_groups(analysis))["inertia"]
    tensor = next(item for item in items if item.symbol == "J")
    assert tensor.value_tex.count(r"\\") == 2, "a 3x3 has two row separators"
    assert "bmatrix" in tensor.value_tex
    for row in analysis.vehicle.inertia_kgm2:
        for value in row:
            assert f"{float(value):g}" in tensor.value_tex


def test_the_nomenclature_defines_every_symbol_the_report_speaks_in():
    """The section exists to answer "what is that letter", so it must answer it."""
    defined = " ".join(tex for _, _, rows in _NOMENCLATURE for tex, _, _, _ in rows)
    for symbol in (
        r"J",
        r"J_{\min}",
        r"J_{\max}",
        r"\omega",
        r"\omega_{\mathrm{tipoff}}",
        r"\omega_n",
        r"\zeta",
        r"h",
        r"h_{\mathrm{envelope}}",
        r"r_{\mathrm{in}}",
        r"r_{\mathrm{out}}",
        r"\tau",
        r"\tau_{\mathrm{sec}}",
        r"\tau_{\mathrm{cyc}}",
        r"B",
        r"|B|_{\min}",
        r"m_{\mathrm{res}}",
        r"\eta",
        r"d_{\mathrm{duty}}",
        r"T_{\mathrm{orbit}}",
        r"T_{\mathrm{desat}}",
        r"\omega_o",
        r"\sigma",
        r"\Delta t",
        r"\xi",
        r"W",
        r"\sigma_{\min}",
        r"\sigma_{\max}",
    ):
        assert symbol in defined, f"nomenclature does not define {symbol}"


def test_the_nomenclature_names_the_flight_parameter_where_there_is_one():
    """The symbol-to-config-key link is the row a reviewer actually needs."""
    parameters = {
        parameter for _, _, rows in _NOMENCLATURE for *_, parameter in rows if parameter
    }
    for key in (
        "MomentumEnvelopeNms",
        "DetumbleExitRadps",
        "BdotGainNms",
        "PidKpNmPerRad",
        "PidKdNmPerRadps",
    ):
        assert any(key in p for p in parameters), f"no nomenclature row names {key}"


def test_a_formula_with_no_latex_degrades_to_plain_text_not_to_broken_math():
    """The deliberate failure mode: legible, obviously untypeset, never a guess."""
    rendered = tex_html("some new formula nobody wrote LaTeX for")
    assert rendered == '<span class="m">some new formula nobody wrote LaTeX for</span>'
    assert "data-tex" not in rendered


def test_a_formula_with_latex_keeps_its_plain_text_for_the_no_script_reader():
    """The ASCII is the element's content; the script replaces it, or does not."""
    rendered = tex_html("Kp = J * wn^2", r"K_p = J\,\omega_n^{2}")
    assert 'data-tex="K_p = J\\,\\omega_n^{2}"' in rendered
    assert math_html("Kp = J * wn^2") in rendered


def test_the_markup_escapes_latex_that_would_otherwise_close_the_attribute():
    """Formula strings are config-adjacent; none of them may open a tag."""
    rendered = tex_html("x", r'\text{"><script>alert(1)</script>}')
    assert "<script>" not in rendered
    assert "&quot;&gt;&lt;script&gt;" in rendered


def test_the_inertia_matrix_keeps_its_plain_form_beside_the_latex():
    """The matrix is a value in a provenance row, so it sets inline with its units.

    Display mode would break the line and strand the ``kg.m^2`` under it; what
    the row needs is the datasheet reading, value and units on one line.
    """
    rendered = tex_value(r"J = \begin{bmatrix} 1 & 0 \end{bmatrix}", "diag(1)")
    assert "data-display" not in rendered
    assert "diag(1)" in rendered
    assert "bmatrix" in rendered


def test_a_bare_symbol_falls_back_to_something_a_reader_can_still_read():
    """The nomenclature has no ASCII source, so the fallback is derived from the LaTeX.

    Derivation in this direction only ever produces text, so the worst case is a
    plainly-set symbol — never mathematics that means something else.
    """
    assert "ω<sub>tipoff</sub>" in tex_symbol(r"\omega_{\mathrm{tipoff}}")
    assert "|B|<sub>min</sub>" in tex_symbol(r"|B|_{\min}")


@pytest.mark.skipif(shutil.which("node") is None, reason="no Node to run KaTeX with")
def test_every_latex_form_sets_in_the_katex_that_ships_with_the_page(analysis, report):
    """The library that renders the page validates the strings, at the same version."""
    items = _every_tex(analysis, report)
    assert items, "the analysis emitted no LaTeX at all"
    result = subprocess.run(
        [
            "node",
            "-e",
            _VALIDATE_JS,
            str((_VENDOR / "katex.min.js").resolve()),
            json.dumps(items),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    failures = json.loads(result.stdout)
    assert not failures, "KaTeX refuses:\n" + "\n".join(failures)
