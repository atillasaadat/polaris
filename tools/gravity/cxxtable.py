"""Emit the flight-side low-degree geopotential coefficient header from a `.gfc`.

Ground-side only (design doc §3.7, §8.3). The committed reference datum stays the
native ICGEM `.gfc` (``tests/golden/EGM2008_to200.gfc``); this module *derives*
a small C++ header from it so the onboard orbit filter can carry a truncated
geopotential without file I/O, which flight code does not have (§3.6).

Two conversions happen here, and both are deliberate:

1. **Truncation** to a low degree/order. 8x8 is 45 coefficient pairs; the full
   committed degree-200 window is 20301.
2. **De-normalization.** EGM2008 ships the 4-pi fully-normalized
   :math:`\\bar C_{nm}, \\bar S_{nm}` that pair with normalized Legendre
   functions. The flight evaluator uses the *unnormalized* Cunningham V/W
   recursion (Montenbruck & Gill 2000, §3.2.4), which is the standard low-degree
   onboard formulation and needs unnormalized coefficients, so the conversion is
   done once here on the ground rather than every cycle in flight:

   .. math::

      C_{nm} = \\bar C_{nm}\\,\\sqrt{\\frac{(2-\\delta_{0m})(2n+1)(n-m)!}{(n+m)!}}

   Sanity anchor: :math:`n=2, m=0` gives :math:`\\sqrt5`, and
   :math:`\\bar C_{20} = -4.8417\\times10^{-4}` maps to :math:`C_{20} = -J_2 =
   -1.0826\\times10^{-3}`, which ``_self_check`` asserts.

   De-normalizing is safe *only because the degree is low*. The factor grows
   like :math:`\\sqrt{(2n+1)(n-m)!/(n+m)!}`, which underflows past degree ~40 —
   the same overflow that makes the unnormalized recursion unusable for the
   truth sim's degree-200 field, where ``sim/world/gravity_field`` uses the
   normalized Gottlieb recursion instead. ``emit_header`` refuses above
   :data:`MAX_SAFE_DEGREE`.

The generated header is a committed derived artifact, and
``tests/tools/test_gravity_cxxtable.py`` regenerates it from the committed `.gfc`
and asserts the *coefficients* match, so it cannot drift from its source
unnoticed — the same discipline as the GMAT golden fixtures (§23.1). The test
compares parsed numbers rather than bytes because clang-format owns the
committed file's whitespace and the generator does not.

References:
 - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.2.4 (Cunningham V/W
   recursion and the unnormalized coefficients it consumes). [montenbruck2000]
 - Pavlis et al., "The development and evaluation of EGM2008", JGR 117, 2012.
   [pavlis2012]
 - Barthelmes & Forste, "The ICGEM `gfc` data format", GFZ. [icgemformat]
"""

from __future__ import annotations

import math

# Above this the de-normalization factorials lose too much precision to be worth
# trusting in double, and the unnormalized recursion the coefficients feed is
# itself unstable there. The onboard model lives an order of magnitude below it.
MAX_SAFE_DEGREE = 20

# `.clang-format`'s ColumnLimit. Emitting inside it keeps the generator and the
# formatter from fighting over the committed file.
_COLUMN_LIMIT = 100


def _float(token: str) -> float:
    """Parse a `.gfc` numeric token, tolerating the Fortran `D` exponent."""
    return float(token.replace("D", "E").replace("d", "e"))


def parse_header(text: str) -> dict[str, float]:
    """Read `earth_gravity_constant` and `radius` from the `.gfc` header block.

    Both are the model's own values, not WGS84's. They must travel with the
    coefficients: a field evaluated with a GM the coefficients were not solved
    with is inconsistent at the level the truncation itself costs.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "earth_gravity_constant" and len(parts) >= 2:
            out["gm"] = _float(parts[1])
        elif parts[0] == "radius" and len(parts) >= 2:
            out["radius"] = _float(parts[1])
        elif parts[0] == "end_of_head":
            break
    missing = {"gm", "radius"} - out.keys()
    if missing:
        raise ValueError(f"`.gfc` header is missing {sorted(missing)}")
    return out


def denormalization_factor(n: int, m: int) -> float:
    """N_nm such that C_nm = N_nm * Cbar_nm (see the module docstring)."""
    delta = 1.0 if m == 0 else 0.0
    log = (
        math.log(2.0 - delta)
        + math.log(2 * n + 1)
        + math.lgamma(n - m + 1)
        - math.lgamma(n + m + 1)
    )
    return math.exp(0.5 * log)


def parse_coefficients(
    text: str, max_degree: int
) -> dict[tuple[int, int], tuple[float, float]]:
    """Unnormalized ``{(n, m): (C_nm, S_nm)}`` for all degrees <= max_degree.

    ``C_00`` is forced to 1 (the point-mass term) regardless of what the file
    carries, matching ``sim/world/egm2008.cpp``.
    """
    coeffs: dict[tuple[int, int], tuple[float, float]] = {}
    in_head = True
    for line in text.splitlines():
        parts = line.split()
        if in_head:
            if parts[:1] == ["end_of_head"]:
                in_head = False
            continue
        if len(parts) < 5 or parts[0] not in ("gfc", "gfct"):
            continue
        n, m = int(parts[1]), int(parts[2])
        if n > max_degree or m > n:
            continue
        factor = denormalization_factor(n, m)
        coeffs[(n, m)] = (_float(parts[3]) * factor, _float(parts[4]) * factor)
    if not coeffs:
        raise ValueError("no coefficient lines found — not a `.gfc`?")
    coeffs[(0, 0)] = (1.0, 0.0)
    return coeffs


def emit_header(text: str, max_degree: int, source: str) -> str:
    """Render the committed C++ header for the `.gfc` in *text*.

    *source* names the committed `.gfc` this was derived from and is written into
    the header so the provenance travels with the numbers.
    """
    if not 2 <= max_degree <= MAX_SAFE_DEGREE:
        raise ValueError(
            f"max_degree must be in [2, {MAX_SAFE_DEGREE}], got {max_degree}"
        )
    head = parse_header(text)
    coeffs = parse_coefficients(text, max_degree)

    lines = [
        "#ifndef POLARIS_GNC_EGM2008_LOW_DEGREE_HPP",
        "#define POLARIS_GNC_EGM2008_LOW_DEGREE_HPP",
        "",
        "/// @file",
        f"/// @brief EGM2008 geopotential truncated to degree/order {max_degree},",
        "/// UNNORMALIZED, for the onboard orbit filter's force model (design doc §8.3).",
        "///",
        "/// GENERATED — do not edit. Regenerate with:",
        "///",
        "///     PYTHONPATH=tools uv run python -m gravity cxx-header",
        f"///       --input {source}",
        f"///       --max-degree {max_degree}",
        "///       --out lib/gnc/egm2008_low_degree.hpp",
        "///",
        "/// (one shell command; the arguments are split across lines here because a",
        "/// trailing backslash inside a `//` comment trips -Wcomment, and this file is",
        "/// compiled with -Werror.)",
        "///",
        f"/// Derived from the committed `{source}` (design doc §3.7: the reference",
        "/// datum stays the native ICGEM `.gfc`; this is a derivation *from* it, not a",
        "/// substitute for it). `tests/tools/test_gravity_cxxtable.py` regenerates this",
        "/// file and asserts the coefficients match, so the two cannot drift apart.",
        "///",
        "/// The coefficients are **unnormalized** — de-normalized on the ground by",
        "/// `tools/gravity/cxxtable.py` — because the flight evaluator",
        "/// (@ref polaris::gnc::geopotentialAcceleration) uses the unnormalized",
        "/// Cunningham V/W recursion (Montenbruck & Gill §3.2.4 [montenbruck2000]).",
        "/// `kC[0][0] = 1` is the point-mass term.",
        "///",
        "/// `kGm` and `kReferenceRadius` are the **model's own** values from the `.gfc`",
        "/// header, not WGS84's. They must be used together with these coefficients:",
        "/// the coefficients were solved with this GM and this radius, and pairing them",
        "/// with a generic constant reintroduces an error comparable to the truncation",
        "/// the table exists to remove.",
        "",
        "#include <cstddef>",
        "",
        "namespace polaris::gnc::egm2008 {",
        "",
        "/// Truncation degree and order of the table below.",
        f"inline constexpr int kMaxDegree = {max_degree};",
        "",
        "/// Gravitational parameter the coefficients were solved with [m^3/s^2].",
        f"inline constexpr double kGm = {head['gm']!r};",
        "",
        "/// Reference radius the coefficients are scaled to [m].",
        f"inline constexpr double kReferenceRadius = {head['radius']!r};",
        "",
        "/// Unnormalized C_nm, row n, column m; entries with m > n are zero.",
        "inline constexpr double kC[kMaxDegree + 1][kMaxDegree + 1] = {",
    ]
    lines += _emit_table(coeffs, max_degree, index=0)
    lines += [
        "};",
        "",
        "/// Unnormalized S_nm, row n, column m; entries with m > n (and m = 0) are zero.",
        "inline constexpr double kS[kMaxDegree + 1][kMaxDegree + 1] = {",
    ]
    lines += _emit_table(coeffs, max_degree, index=1)
    lines += [
        "};",
        "",
        "}  // namespace polaris::gnc::egm2008",
        "",
        "#endif  // POLARIS_GNC_EGM2008_LOW_DEGREE_HPP",
        "",
    ]
    return "\n".join(lines)


def _emit_table(
    coeffs: dict[tuple[int, int], tuple[float, float]], max_degree: int, index: int
) -> list[str]:
    """One brace-enclosed row per degree, wrapped inside the project column limit.

    Values are ``repr``-formatted so they round-trip through double exactly. The
    wrapping is greedy to `.clang-format`'s 100-column limit rather than one
    value per line: clang-format bin-packs braced initializers, so emitting
    something it would reflow just guarantees the generator and the formatter
    disagree forever. Whitespace is not the contract anyway — the drift test
    compares the parsed *numbers*, not the bytes.
    """
    rows: list[str] = []
    for n in range(max_degree + 1):
        values = [coeffs.get((n, m), (0.0, 0.0))[index] for m in range(max_degree + 1)]
        tokens = [f"{v!r}," for v in values]
        tokens[-1] = tokens[-1].rstrip(",")
        suffix = f"}},  // n = {n}"
        line = f"    {{{tokens[0]}"
        for index_, token in enumerate(tokens[1:], start=1):
            # The closing brace and the degree comment ride on the final line, so
            # the last token has to budget for them or the row overruns by exactly
            # the suffix — which is how this was wrong the first time.
            tail = len(suffix) if index_ == len(tokens) - 1 else 0
            if len(line) + 1 + len(token) + tail > _COLUMN_LIMIT:
                rows.append(line)
                line = f"     {token}"
            else:
                line = f"{line} {token}"
        rows.append(f"{line}{suffix}")
    return rows


def _self_check() -> None:
    """Pin the de-normalization against the two values it is easiest to get wrong."""
    # n=2, m=0: factor sqrt(5), and EGM2008's Cbar_20 must land on -J2.
    assert abs(denormalization_factor(2, 0) - math.sqrt(5.0)) < 1e-15
    c20 = -4.84169317366974e-04 * denormalization_factor(2, 0)
    assert abs(c20 + 1.0826e-3) < 1e-7, c20
    # n=0, m=0 is unity: no normalization applies to the point-mass term.
    assert abs(denormalization_factor(0, 0) - 1.0) < 1e-15
    # A sectoral term, where the (2 - delta_0m) branch is the thing under test.
    assert abs(denormalization_factor(2, 2) - math.sqrt(2.0 * 5.0 / 24.0)) < 1e-15

    gfc = (
        "earth_gravity_constant 3.986004415E+14\nradius 6378136.3\n"
        "max_degree 2190\nnorm fully_normalized\nend_of_head ====\n"
        "gfc 0 0 1.0d0 0.0d0\ngfc 2 0 -4.84169317366974e-04 0.0\n"
        "gfc 2 2 2.43938e-06 -1.40027e-06\ngfc 3 0 9.57e-07 0.0\n"
    )
    head = parse_header(gfc)
    assert head["radius"] == 6378136.3, head
    parsed = parse_coefficients(gfc, 2)
    assert (3, 0) not in parsed, "degree filter did not apply"
    assert parsed[(0, 0)] == (1.0, 0.0)
    text = emit_header(gfc, 2, "tests/golden/EGM2008_to200.gfc")
    assert "kMaxDegree = 2" in text
    assert text.endswith("\n")
    print("gravity.cxxtable self-check: ok")


if __name__ == "__main__":
    _self_check()
