"""pytest <-> sphinx-needs traceability collector (design doc §22.2).

Tests declare the requirement(s) they verify::

    @pytest.mark.verifies("REQ-ACTL-014")
    def test_desat_threshold(record_property):
        margin = run_case()
        record_property("margin_pct", margin)   # optional, for quantitative reqs
        assert margin >= 20.0

After the session this writes ``docs/_generated/verif_pytest.json`` in the
sphinx-needs external-needs format, so each test becomes a ``verified by``
back-link on its requirement(s) in the RVTM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "docs" / "_generated" / "verif_pytest.json"

_collected: dict[str, dict] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "verifies(*req_ids): requirement ID(s) this test verifies"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    marker = item.get_closest_marker("verifies")
    if marker is None:
        return
    req_ids = [str(r) for r in marker.args]
    props = dict(getattr(report, "user_properties", []))
    need_id = "TEST_PY_" + item.nodeid.replace("::", "__").replace("/", "_").replace(
        ".", "_"
    )
    _collected[need_id] = {
        "id": need_id,
        "type": "test",
        "title": item.name,
        "status": "passed" if report.passed else "failed",
        "verifies": req_ids,
        "margin_achieved": str(props.get("margin_pct", "")),
        "content": item.nodeid,
    }


def pytest_sessionfinish(session, exitstatus):
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "current_version": "1.0",
        "project": "Polaris pytest verification",
        "versions": {"1.0": {"needs": _collected}},
    }
    _OUT.write_text(json.dumps(data, indent=2))
