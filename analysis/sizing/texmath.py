"""Typeset the report's formulae as LaTeX, offline, with no JS payload.

The console strings are ASCII written for a fixed-width terminal
(``Kp = J * wn^2``). :mod:`analysis.sizing.mathfmt` sets those as Unicode with
``<sub>``/``<sup>``, which is honest but is not typesetting: a fraction stays a
slash and a square root stays a function call. This module renders the same
formulae as real mathematics.

The renderer is **matplotlib mathtext**, which is already a dependency and
already ships the glyphs. Each formula becomes a tight, transparent SVG embedded
as a ``data:`` URI, so the page keeps the one property it must keep: it fetches
nothing at runtime. No MathJax, no KaTeX, no web font.

Baseline alignment
------------------
mathtext reports a box's width, height and **depth** (how far it hangs below the
baseline). The figure is sized to the box exactly and the text placed at the
baseline inside it, so the image can be aligned to the surrounding line with
``vertical-align: -<depth>em``. Sizes are emitted in ``em`` at a 16 px reference,
so an equation set inside a footnote shrinks with it.

Why a lookup table and not a parser
-----------------------------------
Heuristically translating ASCII math to LaTeX is a guessing machine that fails
silently on the one formula nobody checked. :data:`TEX` maps each formula string
this report actually emits to the LaTeX a reviewer should see, and
``tests/analysis/test_sizing_texmath.py`` asserts the table covers every formula
the analysis produces, so an edit upstream is caught rather than degraded
quietly. Anything unmapped, and any mathtext failure, falls back to
:func:`analysis.sizing.mathfmt.math_html`.

References
----------
Design doc §12 (analysis tools); ``analysis/CLAUDE.md`` (the reporting
convention: this is rendering, never a verdict).
"""

from __future__ import annotations

import base64
import functools
import html as _html
from io import BytesIO

import matplotlib
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser

from analysis.sizing.mathfmt import math_html

#: Reference font size [px] the glyphs are rendered at. The emitted geometry is
#: divided by it, so the markup is in ``em`` and scales with its context.
_SIZE_PX = 16.0

#: Ink colour, matching ``--ink`` in the page stylesheet. The page is light
#: unconditionally (see :mod:`analysis.sizing.html`), so a fixed colour is
#: correct rather than a limitation.
_INK = "#16191d"

#: Every formula string the sizing report emits, and the mathematics it means.
#: Keys are matched exactly, and as a **prefix** of a criterion note, since the
#: measuring modules write notes as "<formula> = <value> (<inputs>)".
TEX: dict[str, str] = {
    # -- derived flight parameters (analysis/sizing/parameters.py) --
    "Kp = J * wn^2": r"K_p = J\,\omega_n^2",
    "Kd = 2 * zeta * wn * J": r"K_d = 2\,\zeta\,\omega_n\,J",
    "h_limit = MAX_SISO_COUPLING_RATIO * J_min * w_crossover": (
        r"h_{\mathrm{limit}} = \mathrm{MAX\_SISO\_COUPLING\_RATIO}"
        r"\cdot J_{\min}\,\omega_{\mathrm{c}}"
    ),
    "enter = 0.5 * MomentumEnvelopeNms": (
        r"h_{\mathrm{enter}} = 0.5\,h_{\mathrm{envelope}}"
    ),
    "exit = 0.3 * enter": r"h_{\mathrm{exit}} = 0.3\,h_{\mathrm{enter}}",
    "max(f * h_usable / J_max, sigma*sqrt(2)/(dt*|B|_min))": (
        r"\omega_{\mathrm{exit}} = \max\left("
        r"\frac{f\,h_{\mathrm{usable}}}{J_{\max}},\;"
        r"\frac{\sigma\sqrt{2}}{\Delta t\,|B|_{\min}}\right)"
    ),
    "k >= 2 * omega_o * (1 + sin xi) * J_min": (
        r"k \geq 2\,\omega_o\,(1+\sin\xi)\,J_{\min}"
    ),
    # -- disturbance-torque budget (analysis/sizing/disturbances.py) --
    "3*mu/(2*R^3) * |I_max - I_min|": (
        r"\frac{3\mu}{2R^3}\,\left|I_{\max}-I_{\min}\right|"
    ),
    "0.5 * rho * V^2 * Cd * A * |d_cp|": (
        r"\frac{1}{2}\,\rho\,V^2\,C_d\,A\,\left|d_{cp}\right|"
    ),
    "(Phi/c) * A * Cr * |d_cp|": r"\frac{\Phi}{c}\,A\,C_r\,\left|d_{cp}\right|",
    "|m_res| * |B|_max": r"\left|m_{\mathrm{res}}\right|\,|B|_{\max}",
    # -- momentum sizing drivers (analysis/sizing/wheels.py) --
    "|J * omega_tipoff|": r"\left|J\,\omega_{\mathrm{tipoff}}\right|",
    "|J * DetumbleExitRadps|": r"\left|J\,\omega_{\mathrm{exit}}\right|",
    "0.707 * tau_cyclic * T_orbit / 4": (
        r"0.707\,\tau_{\mathrm{cyc}}\,\frac{T_{\mathrm{orbit}}}{4}"
    ),
    "tau_secular * T_desat": r"\tau_{\mathrm{sec}}\,T_{\mathrm{desat}}",
    "|J * omega_slew|": r"\left|J\,\omega_{\mathrm{slew}}\right|",
    # -- criterion notes that open with a formula --
    "k >= 2*omega_o*(1 + sin xi)*J_min": (r"k \geq 2\,\omega_o\,(1+\sin\xi)\,J_{\min}"),
    "eta * m_in * |B|_min * duty": (
        r"\eta\,m_{\mathrm{in}}\,|B|_{\min}\,d_{\mathrm{duty}}"
    ),
    "floor = sigma*sqrt(2)/(dt*|B|)": (
        r"\omega_{\mathrm{floor}} = \frac{\sigma\sqrt{2}}{\Delta t\,|B|}"
    ),
}

#: Longest first, so ``k >= 2 * omega_o * ...`` is preferred over any prefix of
#: it when a note is matched.
_KEYS_BY_LENGTH = tuple(sorted(TEX, key=len, reverse=True))


@functools.lru_cache(maxsize=256)
def _svg(tex: str) -> str | None:
    """One LaTeX string as an ``<img>`` element, or ``None`` if it will not set.

    Cached: the same formula appears on several rows, and each render is a
    matplotlib figure.
    """
    try:
        matplotlib.use("Agg", force=False)
        body = f"${tex}$"
        width, height, depth, _, _ = MathTextParser("path").parse(
            body, dpi=72, prop=FontProperties(size=_SIZE_PX)
        )
        if not (width > 0.0 and height > 0.0):
            return None
        figure = Figure(figsize=(width / 72.0, height / 72.0))
        figure.patch.set_alpha(0.0)
        figure.text(
            0.0, depth / height, body, fontsize=_SIZE_PX, color=_INK, va="baseline"
        )
        buffer = BytesIO()
        # ``Date: None`` keeps the bytes reproducible run to run.
        figure.savefig(buffer, format="svg", transparent=True, metadata={"Date": None})
    except Exception:  # noqa: BLE001 - a formula must never break the report
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<img class="tex" src="data:image/svg+xml;base64,{encoded}" '
        f'style="height:{height / _SIZE_PX:.4f}em;'
        f'vertical-align:{-depth / _SIZE_PX:.4f}em" '
        f'alt="{_html.escape(tex, quote=True)}">'
    )


def tex_html(text: object) -> str:
    """Set a known formula as typeset mathematics; fall back to Unicode.

    Parameters
    ----------
    text : object
        A formula string as the report objects carry it, e.g. ``Kp = J * wn^2``.

    Returns
    -------
    str
        An ``<img>`` carrying the rendered equation, or the
        :func:`~analysis.sizing.mathfmt.math_html` rendering when the formula is
        not in :data:`TEX` or mathtext refuses it.

    Examples
    --------
    >>> tex_html("Kp = J * wn^2").startswith('<img class="tex"')
    True
    >>> tex_html("not a known formula")
    'not a known formula'
    """
    formula = str(text).strip()
    tex = TEX.get(formula)
    rendered = _svg(tex) if tex else None
    return rendered if rendered is not None else math_html(text)


def split_leading_formula(text: str) -> tuple[str, str]:
    """Split a note into the formula it opens with and the rest.

    ``"tau_secular * T_desat = 0.004 N.m.s (...)"`` becomes the typeset formula
    and ``"= 0.004 N.m.s (...)"``, so a criterion row can show the equation as
    mathematics and its evaluated numbers as ordinary text beside it.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple of str
        ``(formula_html, remainder)``. ``formula_html`` is empty when the note
        does not open with a known formula, in which case the remainder is the
        whole note.
    """
    stripped = text.strip()
    for key in _KEYS_BY_LENGTH:
        if stripped.startswith(key):
            rendered = tex_html(key)
            if rendered.startswith("<img"):
                return rendered, stripped[len(key) :].strip()
    return "", stripped
