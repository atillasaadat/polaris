"""GMAT drift check — the one test that needs the GMAT binary (REQ-VV-002).

Regenerates ``tests/golden/time_scales.json`` with ``GmatConsole`` and asserts the
committed fixture still agrees with GMAT within tolerance. It is SKIPPED when GMAT
is absent — the normal case, including every PR. Enable it by setting
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "golden" / "time_scales.json"


def _gmat_console() -> str | None:
    return os.environ.get("GMAT_CONSOLE") or shutil.which("GmatConsole")


@pytest.mark.verifies("REQ-VV-002")
@pytest.mark.skipif(
    _gmat_console() is None,
    reason="GMAT not installed (set GMAT_CONSOLE or add GmatConsole to PATH)",
)
def test_committed_fixture_agrees_with_gmat():
    committed = json.loads(_FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = regenerate_time_scales_fixture(_gmat_console(), tmp)
    drift = compare_time_scales(committed, regenerated)
    assert not drift, "GMAT disagrees with committed fixture:\n" + "\n".join(drift)
