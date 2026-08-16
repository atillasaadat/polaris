"""The sizing report's own presentation vocabulary.

The generic typography — mathematics, SI prefixes, sentence casing, signed
margins — lives in :mod:`analysis.common.mathfmt` and is shared by every report.
What stays here is the part that is *about sizing*: the ``D``/``M`` short codes
this analysis's criteria are named with, and the provenance keys its report
groups configuration under. Neither means anything to another package, and a
report that needs its own vocabulary should keep it beside itself in the same
way rather than growing the shared module.

The generic names are re-exported so a caller reaching for one formatter does not
have to know which of the two modules it happens to live in.

References
----------
Design doc §12 (analysis tools); ``analysis/CLAUDE.md`` (the reporting
convention — the structured report stays the verdict, this is rendering only).
"""

from __future__ import annotations

import re

from analysis.common.mathfmt import (
    UnitScale,
    _num,
    math_html,
    percent,
    sentence_case,
    signed,
    unit_html,
    unit_scale,
)

__all__ = [
    "UnitScale",
    "_num",
    "criterion_label",
    "math_html",
    "percent",
    "provenance_label",
    "sentence_case",
    "signed",
    "unit_html",
    "unit_scale",
]

#: A short code at the head of the name it abbreviates: ``D1``, ``D1b``, ``M3``.
#: Matched with the word that follows so the two can be set apart.
_SHORT_CODE = re.compile(r"\b([DM]\d[a-z]?)\s+(\w)")


def criterion_label(name: str) -> str:
    """Set a criterion's short code apart from the words that expand it.

    ``"M1 desaturation authority"`` becomes ``"M1 · Desaturation authority"``.
    The codes are this report's own and a reader meeting ``D1b`` in a row has no
    way to expand it; the expansion is already in the name, so all this does is
    stop the two running together as one phrase. The definition list at the head
    of each family says what the code *demands*, and the Nomenclature section
    defines the symbols — neither is duplicated here.

    Parameters
    ----------
    name : str
        The criterion's name as its measuring module wrote it.

    Returns
    -------
    str
    """
    return _SHORT_CODE.sub(lambda m: f"{m.group(1)} · {m.group(2).upper()}", str(name))


#: Provenance keys as a reader should see them. The report's keys are terse
#: console labels; ``"bdot floor"`` in particular has no mechanical title-casing
#: that produces ``"B-dot noise floor"``.
_PROVENANCE_LABELS = {
    "wheels": "Reaction wheels",
    "usable": "Usable momentum",
    "rods": "Magnetorquers",
    "inertia": "Inertia and mass",
    "orbit": "Orbit",
    "disturbance": "Disturbance torque",
    "bdot floor": "B-dot noise floor",
}


def provenance_label(key: str) -> str:
    """A provenance key as a document label; unknown keys are sentence-cased.

    Parameters
    ----------
    key : str

    Returns
    -------
    str
    """
    return _PROVENANCE_LABELS.get(key, sentence_case(key))
