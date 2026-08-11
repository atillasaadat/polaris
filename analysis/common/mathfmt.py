"""Presentation formatting for any HTML analysis report: math, units, casing.

A report's strings are written for a fixed-width console — ``Kp = J * wn^2``,
``N.m.s``, ``k >= 2*omega_o*(1 + sin xi)*J_min``, criterion names in lower case.
That is the right rendering for :meth:`AnalysisReport.format_text`, and it stays
exactly as it is: the plain-text report is the record, and the tests assert on
it. This module is the *other* rendering — the same strings set as a reviewer
would expect to see them in a design-review document.

Shared, because the typography is not any one analysis's
--------------------------------------------------------
This module knows about SI units, Greek letters and sentences; it knows nothing
about wheels, magnetorquers or orbit filters. A package's own vocabulary — the
short codes its criteria use, the labels its provenance keys deserve — stays
with that package (:mod:`analysis.sizing.mathfmt` is the worked example), so
adding a report cannot quietly add its private nouns to everybody else's
namespace.

Why Unicode and ``<sub>``/``<sup>`` rather than MathJax or KaTeX
---------------------------------------------------------------
The page is one self-contained offline file (``tests/analysis/test_sizing_html.py``
asserts it fetches nothing). A typesetting engine costs hundreds of kilobytes
plus web fonts and buys nothing here: every formula in this report is a product
of subscripted symbols, and ``ω<sub>n</sub>²`` is exact, tiny, and still
copy-pasteable.

Prefixes are chosen per family, never per cell
----------------------------------------------
:func:`unit_scale` picks one SI prefix for a whole block of quantities in the
same units — a criterion family, a provenance group, one figure's axis — so
``0.0072 N·m·s`` reads as ``7.2 mN·m·s`` and everything it is compared against
reads in the same unit. Scaling each cell to its own magnitude would make a
column unreadable and invite the mis-comparison the prefix exists to prevent.
The scaling is presentation only: the report objects and
:meth:`AnalysisReport.format_text` keep SI base units.

Conservative by construction
----------------------------
:func:`math_html` **escapes first** and then substitutes only tokens it
recognises, in a single pass over the escaped text. Anything it does not
recognise is passed through byte-for-byte rather than guessed at, so the failure
mode is an unconverted ``sqrt`` rather than a corrupted formula — and because
substitution runs on already-escaped text and emits only ``<sub>``/``<sup>``,
nothing a config file contains can inject markup.

References
----------
Design doc §12 (analysis tools); ``analysis/CLAUDE.md`` (the reporting
convention — the structured report stays the verdict, this is rendering only).
"""

from __future__ import annotations

import html as _html
import math
import re


def _num(value: float, digits: int = 4) -> str:
    """A number a reader can scan, across nine orders of magnitude.

    Fixed-point where a reader has intuition for it, scientific notation
    outside that band, and the report's own words for the non-finite cases.

    Parameters
    ----------
    value : float
    digits : int, optional
        Significant figures.

    Returns
    -------
    str
    """
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "\u221e" if value > 0 else "\u2212\u221e"
    if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e5):
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


#: Greek names spelled out in the console strings, and their letters. Applied to
#: a whole token only, so ``etaFoo`` and ``theta_bar`` behave predictably.
_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "Δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "Phi": "Φ",
    "psi": "ψ",
    "omega": "ω",
    "Omega": "Ω",
}

#: Tokens whose conventional rendering is not what the generic rules would give.
_TOKENS = {
    "Kp": "K<sub>p</sub>",
    "Kd": "K<sub>d</sub>",
    "Ki": "K<sub>i</sub>",
    "wn": "ω<sub>n</sub>",
    "omega_o": "ω₀",
    "w_c": "ω<sub>c</sub>",
    "w_crossover": "ω<sub>crossover</sub>",
    "uT": "µT",
    "uN": "µN",
    "uA": "µA",
    "uPa": "µPa",
}

#: Superscript digits, for exponents a font can render without markup.
_SUPERSCRIPT = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
}

#: Unit symbols recognised in a dotted product (``N.m.s``, ``kg.m^2``). Kept to
#: an explicit list so ``leo_smallsat.yaml`` is never mistaken for a unit.
#: Longest-first where one is a prefix of another (``km`` before ``m``).
_UNIT_ATOM = (
    r"(?:kg|nT|uT|kN|mN|uN|nN|kA|mA|uA|nA|deg|rad|km|mm|Pa|Hz|W|N|m|s|A|T|V|K|C|J|g)"
)
_EXPONENT = r"(?:\^-?\d+)?"

#: One ordered pass. First alternative that matches at a position wins, so the
#: specific rules (``sqrt(``, dotted units) are listed before the generic
#: identifier rule that would otherwise swallow their leading token.
_MATH = re.compile(
    r"(?P<arrow>-&gt;)"
    r"|(?P<ge>&gt;=)"
    r"|(?P<le>&lt;=)"
    r"|(?P<sqrt>\bsqrt\((?P<sqrt_arg>[^()]*)\))"
    r"|(?P<absolute>\|(?P<abs_body>[A-Za-z][A-Za-z0-9]*)\|_(?P<abs_sub>[A-Za-z0-9]+))"
    rf"|(?P<unit>\b{_UNIT_ATOM}{_EXPONENT}(?:\.{_UNIT_ATOM}{_EXPONENT})+\b)"
    r"|(?P<cross>(?<= )x(?= \())"
    r"|(?P<sup>\^(?P<sup_arg>-?\d+|[A-Za-z]))"
    r"|(?P<ident>\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*\b)"
    r"|(?P<times>\*)"
)

#: The alternatives of :data:`_MATH`, in the order they appear in it.
_ALTERNATIVES = (
    "arrow",
    "ge",
    "le",
    "sqrt",
    "absolute",
    "unit",
    "cross",
    "sup",
    "ident",
    "times",
)


def _superscript(text: str) -> str:
    """An exponent as Unicode where every character has a superscript form."""
    if all(character in _SUPERSCRIPT for character in text):
        return "".join(_SUPERSCRIPT[character] for character in text)
    return f"<sup>{text}</sup>"


def _identifier(token: str) -> str:
    """One bare identifier: a Greek letter, a subscripted symbol, or itself."""
    if token in _TOKENS:
        return _TOKENS[token]
    base, _, subscript = token.partition("_")
    if not subscript:
        return _GREEK.get(token, token)
    # Only a *symbol* takes a subscript: one letter, or a Greek name. Anything
    # else with an underscore in it is a name that happens to contain one —
    # MAX_SISO_COUPLING_RATIO is a constant, leo_smallsat is half a file path,
    # and subscripting either is corruption rather than typesetting.
    if base in _GREEK or (len(base) == 1 and base.isalpha()):
        return f"{_GREEK.get(base, base)}<sub>{subscript}</sub>"
    return token


def _substitute(match: re.Match[str]) -> str:
    """Render whichever alternative matched."""
    # Not ``match.lastgroup``: that reports the last *named* group to match,
    # which for ``sqrt`` is the inner ``sqrt_arg``.
    kind = next(name for name in _ALTERNATIVES if match.group(name) is not None)
    if kind == "arrow":
        return "→"
    if kind == "ge":
        return "≥"
    if kind == "le":
        return "≤"
    if kind == "sqrt":
        argument = _MATH.sub(_substitute, match.group("sqrt_arg"))
        # Parentheses only where dropping them would change what binds.
        bare = re.fullmatch(r"[0-9A-Za-z_]*", match.group("sqrt_arg"))
        return f"√{argument}" if bare else f"√({argument})"
    if kind == "absolute":
        return f"|{match.group('abs_body')}|<sub>{match.group('abs_sub')}</sub>"
    if kind == "unit":
        return "·".join(
            re.sub(
                r"^([A-Za-z]+)",
                lambda m: _TOKENS.get(m.group(1), m.group(1)),
                re.sub(r"\^(-?\d+)", lambda m: _superscript(m.group(1)), part),
            )
            for part in match.group("unit").split(".")
        )
    if kind == "cross":
        return "×"
    if kind == "sup":
        return _superscript(match.group("sup_arg"))
    if kind == "times":
        return "·"
    return _identifier(match.group("ident"))


def math_html(text: object) -> str:
    """Escape @p text and set its plain-text math as HTML.

    ``Kp = J * wn^2`` becomes ``K<sub>p</sub> = J · ω<sub>n</sub>²``. Tokens the
    rules do not recognise are passed through unchanged.

    Parameters
    ----------
    text : object
        Any value; stringified, then escaped with :func:`html.escape`
        (``quote=True``) **before** any substitution runs.

    Returns
    -------
    str
        HTML-safe markup containing at most ``<sub>`` and ``<sup>`` elements.

    Examples
    --------
    >>> math_html("k >= 2*omega_o*J_min")
    'k ≥ 2·ω₀·J<sub>min</sub>'
    >>> math_html("a < b & c")
    'a &lt; b &amp; c'
    """
    return _MATH.sub(_substitute, _html.escape(str(text), quote=True))


#: The SI prefixes the report will attach, with the power of ten each names.
#: Deliberately short: a sizing report spans nano to kilo and nothing outside it,
#: and a prefix a reader has to look up costs more than the zeros it saves.
_PREFIXES = ((3, "k"), (0, ""), (-3, "m"), (-6, "u"), (-9, "n"))

#: Units a prefix may be attached to, spelled exactly as the report declares
#: them. An explicit set rather than a rule: ``deg/s`` and ``km`` already carry
#: their own scaling, ``x`` and ``of envelope`` are ratios, and a generic
#: "prefix the first token" rule would happily produce ``mkg`` or ``mdeg/s``.
#: The prefix goes on the leading atom, which is why every member starts with an
#: unprefixed SI symbol.
_PREFIXABLE = frozenset({"N.m.s", "N.m", "A.m^2", "N.m/rad", "N.m/(rad/s)"})


class UnitScale:
    """One display prefix, chosen for a family of quantities and shared by them.

    Rescaling each cell independently would make a column unreadable — 7.2 mN·m·s
    beside 0.049 N·m·s invites exactly the mis-comparison the prefix was meant to
    prevent — so a scale is chosen once for a whole block (a criterion family, a
    provenance group, one figure's axis) and every number in it is set through
    the same instance.

    Attributes
    ----------
    factor : float
        Multiply an SI value by this to get the number displayed.
    units : str
        The display units, prefix attached, in the report's dotted convention
        (``mN.m.s``). Pass through :func:`unit_html` to set it.
    """

    __slots__ = ("factor", "units")

    def __init__(self, factor: float, units: str) -> None:
        self.factor = factor
        self.units = units

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, UnitScale)
            and other.factor == self.factor
            and other.units == self.units
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"UnitScale({self.factor!r}, {self.units!r})"

    def value(self, si: float) -> float:
        """@p si expressed in :attr:`units`."""
        return si * self.factor

    def text(self, si: float, digits: int = 4) -> str:
        """@p si expressed in :attr:`units` and formatted by :func:`_num`."""
        return _num(self.value(si), digits)


def unit_scale(units: object, values: object) -> UnitScale:
    """Pick one SI prefix for a family of quantities in the same units.

    The prefix is chosen from the **median** magnitude present, not the largest
    or the smallest: a family whose values span decades (the magnetorquer torques
    do) has no prefix that flatters all of them, and the median keeps the typical
    row readable while the outliers stay honestly small or large.

    Parameters
    ----------
    units : object
        The declared display units. Anything outside :data:`_PREFIXABLE` is
        returned unscaled, so a ratio or an already-scaled unit passes through.
    values : object
        An iterable of SI values in those units. Zero and non-finite entries are
        ignored; an empty family is returned unscaled.

    Returns
    -------
    UnitScale

    Examples
    --------
    >>> unit_scale("N.m.s", [0.0072, 0.005]).text(0.0072)
    '7.2'
    >>> unit_scale("N.m.s", [0.0072]).units
    'mN.m.s'
    >>> unit_scale("x", [4.678]).units
    'x'
    """
    text = str(units).strip()
    if text not in _PREFIXABLE:
        return UnitScale(1.0, text)
    magnitudes = sorted(
        abs(float(v)) for v in values if math.isfinite(float(v)) and float(v) != 0.0
    )
    if not magnitudes:
        return UnitScale(1.0, text)
    median = magnitudes[len(magnitudes) // 2]
    exponent, prefix = _PREFIXES[-1]
    for power, symbol in _PREFIXES:
        if median * 10.0**-power >= 1.0:
            exponent, prefix = power, symbol
            break
    return UnitScale(10.0**-exponent, prefix + text)


#: Units whose rendering is not a matter of symbols. ``"-"`` is the report's
#: dimensionless marker and reads better as nothing at all.
_UNIT_OVERRIDES = {"-": "", "": "", "x": "×"}


def unit_html(units: object) -> str:
    """Escape and set a units string: ``N.m.s`` → ``N·m·s``, ``-`` → nothing.

    Parameters
    ----------
    units : object
        The criterion's or parameter's declared display units.

    Returns
    -------
    str
        HTML-safe markup; empty for a dimensionless quantity.
    """
    text = str(units).strip()
    if text in _UNIT_OVERRIDES:
        return _UNIT_OVERRIDES[text]
    return math_html(text)


def sentence_case(text: str) -> str:
    """Capitalise the first word, unless it is a symbol or already an identifier.

    ``"usable momentum vs D1b post-B-dot handover"`` becomes ``"Usable momentum
    vs D1b post-B-dot handover"``, while ``"MomentumEnvelopeNms within the SISO
    validity bound"`` and ``"M1 desaturation authority"`` are left alone —
    blanket ``str.title()`` would mangle both.

    The other thing left alone is a line that *opens* with mathematics:
    ``"eta * m_in * |B|_min * duty"`` and ``"k >= 2*omega_o*..."`` start with
    symbols, and capitalising η into ``Eta`` or ``k`` into ``K`` renames a
    quantity rather than tidying a sentence.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    head = text.split(" ", 1)[0]
    if not head or not head.islower() or not head.replace("-", "").isalpha():
        return text
    if head in _GREEK or head in _TOKENS or len(head) == 1:
        return text
    return text[0].upper() + text[1:]


def signed(value: float, digits: int = 4) -> str:
    """A margin with its sign always shown, so the direction is never inferred.

    Parameters
    ----------
    value : float
    digits : int, optional

    Returns
    -------
    str
        ``"n/a"`` for nan, ``"+∞"``/``"-∞"`` for infinities.
    """
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "+∞" if value > 0 else "-∞"
    return ("+" if value >= 0 else "-") + _num(abs(value), digits)


def percent(value: float) -> str:
    """A signed percentage, scaled so five-figure margins stay readable.

    Parameters
    ----------
    value : float
        Percentage, already in percent (not a fraction).

    Returns
    -------
    str
    """
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "+∞%" if value > 0 else "-∞%"
    magnitude = abs(value)
    if magnitude >= 1000.0:
        body = f"{value:,.0f}"
    elif magnitude >= 10.0:
        body = f"{value:.1f}"
    else:
        body = f"{value:.2f}"
    return ("+" + body if value >= 0 else body) + "%"
