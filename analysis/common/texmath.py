"""Typeset the report's formulae with KaTeX, inlined, with no network fetch.

The console strings are ASCII written for a fixed-width terminal
(``Kp = J * wn^2``). :mod:`analysis.common.mathfmt` sets those as Unicode with
``<sub>``/``<sup>``, which is honest but is not typesetting: a fraction stays a
slash and a square root stays a function call. This module renders real
mathematics, from LaTeX the report objects carry themselves.

The LaTeX comes from the source, not from a table
-------------------------------------------------
Every object that carries a ``formula`` carries an optional ``formula_tex``
beside it, written where the object is constructed. Nothing here reverse-engineers
an ASCII string into mathematics, and nothing keys a lookup table off one: a
formula string edited upstream takes its LaTeX with it, or it has none and the
page says so by setting it as plain text. That is the whole architecture — the
previous string-keyed table rotted silently every time a formula was reworded,
and put raw words on a page that claimed to typeset them.

Degrading, deliberately
-----------------------
:func:`tex_html` with no LaTeX returns the
:func:`~analysis.common.mathfmt.math_html` rendering in the page's ordinary math
face. It is legible, it is obviously not typeset, and it is never pseudo-mathematics
assembled from a guess. The same rendering is what a reader with JavaScript
disabled sees for *every* formula: the markup carries the ASCII as its element
content and the LaTeX in ``data-tex``, and the page's script replaces the one with
the other on load.

Self-contained by construction
------------------------------
KaTeX 0.16.11 is vendored under ``vendor/katex/`` (see its ``PROVENANCE.md``).
:func:`katex_assets` emits the stylesheet with the eight WOFF2 faces base64-inlined
into their ``@font-face`` rules, the library, and the render call — so the page
fetches nothing at runtime, which is the property
``tests/analysis/test_sizing_html.py`` exists to keep.

References
----------
Design doc §12 (analysis tools); ``analysis/CLAUDE.md`` (the reporting
convention: this is rendering, never a verdict).
"""

from __future__ import annotations

import base64
import functools
import html as _html
import re
from pathlib import Path

from analysis.common.mathfmt import math_html

#: The vendored library, beside this module so it travels with the package
#: rather than with the working directory.
_VENDOR = Path(__file__).parent / "vendor" / "katex"

#: One ``@font-face`` rule, as the minified stylesheet writes it.
_FONT_FACE = re.compile(r"@font-face\{[^}]*\}")

#: The WOFF2 file a rule asks for, relative to the stylesheet.
_WOFF2 = re.compile(r"url\(fonts/(KaTeX_[A-Za-z0-9-]+)\.woff2\)")

#: The whole ``src`` declaration, whatever formats it lists.
_SRC = re.compile(r"src:[^;}]*")


def _inline_fonts(css: str) -> str:
    """Rewrite ``@font-face`` sources to ``data:`` URIs, dropping what is absent.

    A rule whose WOFF2 file is not vendored is **removed** rather than left
    pointing at ``fonts/…``: a relative URL in an emailed page is a fetch that
    fails, and the page's contract is that it makes none. The glyph then falls
    back to the browser's own face, which is visibly wrong rather than missing.
    """

    def rewrite(match: re.Match[str]) -> str:
        rule = match.group(0)
        found = _WOFF2.search(rule)
        if not found:
            return ""
        path = _VENDOR / "fonts" / f"{found.group(1)}.woff2"
        if not path.is_file():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        source = f"src:url(data:font/woff2;base64,{encoded}) format('woff2')"
        return _SRC.sub(source, rule, count=1)

    return _FONT_FACE.sub(rewrite, css)


@functools.lru_cache(maxsize=1)
def katex_assets() -> tuple[str, str]:
    """The stylesheet and the script the page needs, both self-contained.

    Returns
    -------
    tuple of str
        ``(css, js)``. The CSS is KaTeX's own with the vendored faces inlined;
        the JS is the library followed by the call that renders every marked
        span. Cached: the bytes are the same for every page in a process.
    """
    css = _inline_fonts((_VENDOR / "katex.min.css").read_text(encoding="utf-8"))
    library = (_VENDOR / "katex.min.js").read_text(encoding="utf-8")
    return css, library + _RENDER_JS


#: Replaces each marked span's ASCII content with its typeset form. Guarded on
#: ``katex`` existing and on each render, so one bad formula leaves its own
#: fallback in place and the rest of the page still typesets.
_RENDER_JS = """
;document.addEventListener('DOMContentLoaded', function () {
  if (typeof katex === 'undefined') { return; }
  document.querySelectorAll('span.tex[data-tex]').forEach(function (node) {
    try {
      katex.render(node.getAttribute('data-tex'), node, {
        throwOnError: false, displayMode: node.hasAttribute('data-display'),
        output: 'html', strict: false,
      });
    } catch (error) { /* the ASCII fallback already in the node stands */ }
  });
});
"""


def tex_html(text: object, tex: str = "") -> str:
    """Set a formula as mathematics when it carries LaTeX, as plain text otherwise.

    Parameters
    ----------
    text : object
        The formula as the report objects carry it, e.g. ``Kp = J * wn^2``. It
        is what a reader without JavaScript sees, and what stays in place if
        KaTeX refuses the LaTeX.
    tex : str, optional
        The LaTeX for the same formula, written beside it at its source. Empty
        means the formula has none, and it is set as ordinary styled text
        rather than guessed at.

    Returns
    -------
    str
        HTML-safe markup.

    Examples
    --------
    >>> tex_html("Kp = J * wn^2", r"K_p = J\\,\\omega_n^2")
    '<span class="tex" data-tex="K_p = J\\\\,\\\\omega_n^2">K<sub>p</sub> = J · ω<sub>n</sub>²</span>'
    >>> tex_html("no latex for this one")
    '<span class="m">no latex for this one</span>'
    """
    fallback = math_html(text)
    if not tex:
        return f'<span class="m">{fallback}</span>'
    return f'<span class="tex" data-tex="{_html.escape(tex, quote=True)}">{fallback}</span>'


#: Control sequences that carry no meaning once the LaTeX is reduced to a plain
#: symbol name: spacing, font selection, delimiter sizing.
_TEX_NOISE = re.compile(r"\\(?:mathrm|mathbf|operatorname|left|right|bar|hat|[,;!\s])")


def tex_symbol(tex: str) -> str:
    """A bare symbol, typeset, with a legible plain-text fallback derived from it.

    Used by the nomenclature, whose rows are symbols with no ASCII source string
    to fall back to. Stripping ``\\omega_{\\mathrm{tipoff}}`` down to
    ``omega_tipoff`` and handing that to
    :func:`~analysis.common.mathfmt.math_html` is safe in a way the reverse
    direction never is: it only ever produces *text*, so the worst case is a
    plainly-set symbol rather than mathematics that means something else.

    Parameters
    ----------
    tex : str
        The LaTeX for one symbol.

    Returns
    -------
    str
        HTML-safe markup.
    """
    plain = _TEX_NOISE.sub("", tex)
    plain = plain.replace("\\", "").replace("{", "").replace("}", "")
    return tex_html(plain, tex)


def tex_value(tex: str, alt: object) -> str:
    """A *value* set as mathematics — the inertia tensor, which is a matrix.

    Inline rather than display mode: this is a quantity in a provenance row, not
    an equation in a paragraph, so it has to sit on the same line as its units
    the way it would be written on a datasheet. KaTeX sets a ``bmatrix`` inline
    perfectly well; display mode would break the line and strand the units.

    Parameters
    ----------
    tex : str
        The LaTeX for the value.
    alt : object
        The plain-text form, shown until the script runs and if it never does.

    Returns
    -------
    str
        HTML-safe markup.
    """
    return tex_html(alt, tex)
