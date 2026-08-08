"""The HTML report's presentation formatter (:mod:`analysis.sizing.mathfmt`).

Two properties matter here and they pull in opposite directions. The formatter
has to actually convert the console strings the report produces — otherwise the
page still shows ``wn^2`` — and it has to be **conservative**, because it runs
over config-derived prose: anything it does not recognise must survive
byte-for-byte, and nothing it emits may be markup the caller did not intend.

The strings exercised below are the real ones, copied from what
``analysis.sizing.report`` renders for the reference vehicle.
"""

from __future__ import annotations

import math

import pytest

from analysis.sizing.mathfmt import (
    math_html,
    percent,
    provenance_label,
    sentence_case,
    signed,
    unit_html,
)


# --------------------------------------------------------------------------
# Escaping — the property that must hold before any of the rest is safe
# --------------------------------------------------------------------------


def test_markup_is_escaped_before_any_substitution_runs():
    """A config string cannot become live markup, whatever else happens to it.

    The formatter escapes first and then substitutes on the escaped text, so
    the only way ``<`` reaches the page is as ``&lt;``. This is the test that
    keeps the substitution rules from ever being an injection vector.
    """
    assert math_html('<script>alert("x")</script> & more') == (
        "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; more"
    )
    # The escaped entities themselves must survive the identifier rule: "amp",
    # "lt" and "quot" are bare words to it, and it must leave them alone.
    assert math_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_the_only_elements_emitted_are_sub_and_sup():
    """Substitution may add subscripts and superscripts, and nothing else."""
    rendered = math_html("Kp = J * wn^2 and b^T and |B|_min")
    for fragment in rendered.split("<")[1:]:
        assert fragment.startswith(("sub>", "/sub>", "sup>", "/sup>"))


# --------------------------------------------------------------------------
# The conversions the report actually needs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plain", "expected"),
    [
        # Derived-parameter formulae, verbatim from analysis.sizing.parameters.
        ("Kp = J * wn^2", "K<sub>p</sub> = J · ω<sub>n</sub>²"),
        (
            "Kd = 2 * zeta * wn * J",
            "K<sub>d</sub> = 2 · ζ · ω<sub>n</sub> · J",
        ),
        (
            "k >= 2 * omega_o * (1 + sin xi) * J_min",
            "k ≥ 2 · ω₀ · (1 + sin ξ) · J<sub>min</sub>",
        ),
        (
            "max(f * h_usable / J_max, sigma*sqrt(2)/(dt*|B|_min))",
            "max(f · h<sub>usable</sub> / J<sub>max</sub>, "
            "σ·√2/(dt·|B|<sub>min</sub>))",
        ),
        # Criterion notes.
        (
            "0.707 * tau_cyclic * T_orbit / 4",
            "0.707 · τ<sub>cyclic</sub> · T<sub>orbit</sub> / 4",
        ),
        (
            "eta * m_in * |B|_min * duty",
            "η · m<sub>in</sub> · |B|<sub>min</sub> · duty",
        ),
        ("sigma*sqrt(2)/(dt*|B|)", "σ·√2/(dt·|B|)"),
        ("lambda_min/lambda_max", "λ<sub>min</sub>/λ<sub>max</sub>"),
        # Disturbance-budget formulae.
        (
            "3*mu/(2*R^3) * |I_max - I_min|",
            "3·μ/(2·R³) · |I<sub>max</sub> - I<sub>min</sub>|",
        ),
        (
            "0.5 * rho * V^2 * Cd * A * |d_cp|",
            "0.5 · ρ · V² · Cd · A · |d<sub>cp</sub>|",
        ),
        ("(Phi/c) * A * Cr * |d_cp|", "(Φ/c) · A · Cr · |d<sub>cp</sub>|"),
        # Prose that carries math inline.
        ("omega x (J omega + h)", "ω × (J ω + h)"),
        ("4/(zeta*wn) = 6 s", "4/(ζ·ω<sub>n</sub>) = 6 s"),
    ],
)
def test_the_reports_own_formulae_are_set_as_mathematics(plain, expected):
    """Every plaintext-math string the report emits, and what it becomes."""
    assert math_html(plain) == expected


@pytest.mark.parametrize(
    ("plain", "expected"),
    [
        ("N.m.s", "N·m·s"),
        ("N.m", "N·m"),
        ("kg.m^2", "kg·m²"),
        ("A.m^2", "A·m²"),
        ("uN.m", "µN·m"),
        ("N.m/rad", "N·m/rad"),
        ("N.m/(rad/s)", "N·m/(rad/s)"),
        ("kg/m^3", "kg/m³"),
        ("deg/s", "deg/s"),
        ("rad/s", "rad/s"),
        ("uT", "µT"),
    ],
)
def test_units_are_set_with_middots_and_real_exponents(plain, expected):
    assert unit_html(plain) == expected


def test_a_dimensionless_unit_renders_as_nothing():
    """``-`` is the report's dimensionless marker, not a unit to print."""
    assert unit_html("-") == ""
    assert unit_html(" ") == ""
    assert unit_html("x") == "×"
    # A unit the overrides do not know still goes through the math rules.
    assert unit_html("of envelope") == "of envelope"


def test_input_strings_keep_their_prose_and_convert_their_symbols():
    """The ``inputs`` line mixes numbers, units and an arrow; all three convert."""
    assert math_html(
        "f = 0.5, h_usable = 0.38 N.m.s, J_max = 4.71 kg.m^2 -> 2.31 deg/s"
    ) == (
        "f = 0.5, h<sub>usable</sub> = 0.38 N·m·s, "
        "J<sub>max</sub> = 4.71 kg·m² → 2.31 deg/s"
    )
    assert math_html("|m_res| = 0.01123 A.m^2, |B|_max = 47.3 uT") == (
        "|m<sub>res</sub>| = 0.01123 A·m², |B|<sub>max</sub> = 47.3 µT"
    )


# --------------------------------------------------------------------------
# Conservatism — what must NOT be touched
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "untouched",
    [
        "MomentumEnvelopeNms",
        "DetumbleExitRadps",
        "BdotGainNms",
        "PidKpNmPerRad",
        "REQ-ACTL-009",
        "MAX_SISO_COUPLING_RATIO",
        "AllocMethodSel = 1",
        "config/spacecraft/leo_smallsat.yaml",
        "[avanzini2012]",
        "B-dot",
        "D1b",
        "L-infinity",
        "settling to 2% in about 6 s",
        "1.3x the momentum required by the sizing drivers",
        "thermospheric density at 500 km",
        "6.97e-13",
    ],
)
def test_identifiers_and_prose_the_rules_do_not_own_are_passed_through(untouched):
    """Conservative means unrecognised text survives byte-for-byte.

    ``MAX_SISO_COUPLING_RATIO`` is the motivating case: it is a constant name
    with underscores in it, and a naive subscript rule turns it into
    ``MAX<sub>SISO_COUPLING_RATIO</sub>``. A file path is the other — its dots
    must not become unit separators.
    """
    assert math_html(untouched) == untouched


def test_a_formula_the_rules_do_not_recognise_is_left_alone_not_mangled():
    """The failure mode is an unconverted string, never a corrupted one."""
    exotic = "\\oint_{C} F cdot dr = iint curl F"
    assert math_html(exotic) == exotic


# --------------------------------------------------------------------------
# Casing and labels
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plain", "expected"),
    [
        (
            "usable momentum vs D1b post-B-dot handover",
            "Usable momentum vs D1b post-B-dot handover",
        ),
        (
            "closed-loop bandwidth below the sampling bound",
            "Closed-loop bandwidth below the sampling bound",
        ),
        (
            "wheel torque, guaranteed in every direction",
            "Wheel torque, guaranteed in every direction",
        ),
        ("gravity gradient", "Gravity gradient"),
        # Already-capitalised heads are identifiers and must survive intact.
        (
            "MomentumEnvelopeNms within the SISO validity bound",
            "MomentumEnvelopeNms within the SISO validity bound",
        ),
        (
            "M3 DetumbleExitRadps above the B-dot noise floor",
            "M3 DetumbleExitRadps above the B-dot noise floor",
        ),
        (
            "BdotGainNms above the Avanzini convergence floor",
            "BdotGainNms above the Avanzini convergence floor",
        ),
        ("", ""),
    ],
)
def test_sentence_case_capitalises_words_and_never_identifiers(plain, expected):
    assert sentence_case(plain) == expected


@pytest.mark.parametrize(
    "opens_with_mathematics",
    [
        "eta * m_in * |B|_min * duty",
        "k >= 2*omega_o*(1 + sin xi)*J_min",
        "zeta = 0.7 is what the committed Kp and Kd jointly imply",
        "wn = 0.9 rad/s is 70x below the 62.8 rad/s sample rate",
        "tau_secular * T_desat = 0.000694 N.m.s",
        "0.707 * tau_cyclic * T_orbit / 4",
    ],
)
def test_a_line_that_opens_with_a_symbol_is_not_capitalised(opens_with_mathematics):
    """η is not ``Eta`` and k is not ``K``.

    These are criterion notes and reasoning lines that begin with a quantity
    rather than a word. Sentence case tidies prose; applied here it would rename
    the quantity, which is the one thing a report may not do.
    """
    assert sentence_case(opens_with_mathematics) == opens_with_mathematics


def test_provenance_keys_become_document_labels():
    """``bdot floor`` has no mechanical casing that yields ``B-dot noise floor``."""
    assert provenance_label("bdot floor") == "B-dot noise floor"
    assert provenance_label("wheels") == "Reaction wheels"
    assert provenance_label("usable") == "Usable momentum"
    # An unknown key still gets a readable label rather than a KeyError.
    assert provenance_label("some new key") == "Some new key"


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def test_a_margin_always_carries_its_sign():
    """The sign is the direction past the threshold; it is never inferred."""
    assert signed(0.116) == "+0.116"
    assert signed(-0.116) == "-0.116"
    assert signed(0.0) == "+0"
    assert signed(float("nan")) == "n/a"
    assert signed(float("inf")) == "+∞"
    assert signed(float("-inf")) == "-∞"


def test_percentages_stay_readable_across_five_orders_of_magnitude():
    """44% and 42006% are both real margins in this report and both must scan."""
    assert percent(43.993) == "+44.0%"
    assert percent(1.5) == "+1.50%"
    assert percent(11744.087) == "+11,744%"
    assert percent(-18.5) == "-18.5%"
    assert percent(float("nan")) == "n/a"
    assert percent(math.inf) == "+∞%"
