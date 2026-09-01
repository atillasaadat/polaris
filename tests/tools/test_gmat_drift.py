"""GMAT drift checks — the tests that need the GMAT binary (REQ-VV-002).

Regenerates ``tests/golden/time_scales.json`` and ``tests/golden/gmat_propagation.json``
with ``GmatConsole`` and asserts the committed fixtures still agree with GMAT within
tolerance. They are SKIPPED when GMAT is absent — the normal case, including every PR. Enable it by setting
``$GMAT_CONSOLE`` (or putting ``GmatConsole`` on ``PATH``); the nightly ``golden``
CI lane installs GMAT and does exactly that (design doc §23.1/§23.2).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from gmat.golden import compare_time_scales, regenerate_time_scales_fixture
from gmat.propagation import compare_propagation, regenerate_propagation_fixture

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "golden" / "time_scales.json"
_PROP_FIXTURE = _REPO_ROOT / "tests" / "golden" / "gmat_propagation.json"


def _gmat_console() -> str | None:
    return os.environ.get("GMAT_CONSOLE") or shutil.which("GmatConsole")


_needs_gmat = pytest.mark.skipif(
    _gmat_console() is None,
    reason="GMAT not installed (set GMAT_CONSOLE or add GmatConsole to PATH)",
)


@pytest.mark.verifies("REQ-VV-002")
@_needs_gmat
def test_committed_fixture_agrees_with_gmat():
    committed = json.loads(_FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = regenerate_time_scales_fixture(_gmat_console(), tmp)
    drift = compare_time_scales(committed, regenerated)
    assert not drift, "GMAT disagrees with committed fixture:\n" + "\n".join(drift)


@pytest.mark.verifies("REQ-VV-002")
@_needs_gmat
def test_committed_propagation_fixture_agrees_with_gmat():
    committed = json.loads(_PROP_FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = regenerate_propagation_fixture(_gmat_console(), tmp)
    drift = compare_propagation(committed, regenerated)
    assert not drift, "GMAT disagrees with committed fixture:\n" + "\n".join(drift)


@pytest.mark.verifies("REQ-ODP-003")
@_needs_gmat
def test_committed_sgp4_fixture_agrees_with_gmat():
    """The SGP4/TLE fixture still matches what GMAT's SPICESGP4 recomputes.

    Separate from the numerical-propagation drift check above because it
    exercises a different GMAT subsystem entirely: the ``SPICESGP4`` plugin and
    its TLE reader, not the force-model integrators. A GMAT release that changed
    only its SPICE kernels would move this and leave the other untouched.
    """
    from gmat.tle import FIXTURE_PATH, compare_tle, regenerate_tle_fixture

    committed = json.loads(FIXTURE_PATH.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = regenerate_tle_fixture(_gmat_console(), tmp)
    drift = compare_tle(committed, regenerated)
    assert not drift, "GMAT disagrees with committed SGP4 fixture:\n" + "\n".join(drift)
