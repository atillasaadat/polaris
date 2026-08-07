"""Fixtures for the FreeFlyer cross-validation suite (REQ-VV-006).

The suite needs a licensed FreeFlyer installation this process can drive
(``tools/freeflyer/locate.py``). Where none exists — most CI runners, a
machine without the license seat — every test **skips visibly**; a silent
pass would defeat the point of an independent cross-check.

Set ``POLARIS_FF_QUICK=1`` to run only the fast cases (the pre-push hook
does): the full matrix propagates a GEO day and a Molniya arc and costs a few
minutes of engine time.
"""

from __future__ import annotations

import os

import pytest

from freeflyer import vv
from freeflyer.locate import find_runnable_licensed

#: Cases cheap enough for the pre-push hook (~15 s each).
QUICK_CASES = ("two_body", "zonal_j2", "attitude_spinner")


@pytest.fixture(scope="session")
def ff_install():
    install = find_runnable_licensed()
    if install is None:
        pytest.skip(
            "no runnable licensed FreeFlyer found (POLARIS_FF_DIR to point at "
            "one; see tools/freeflyer/README.md)"
        )
    return install


@pytest.fixture(scope="session")
def ff_case_results(ff_install):
    """Lazily propagate golden cases in FreeFlyer, once each per session."""
    cache: dict[str, list[dict]] = {}

    def run(name: str) -> tuple[dict, list[dict]]:
        case = vv.load_cases()[name]
        if name not in cache:
            cache[name] = vv.run_case(ff_install, case)
        return case, cache[name]

    return run


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("POLARIS_FF_QUICK"):
        return
    skip = pytest.mark.skip(reason="slow FreeFlyer case (POLARIS_FF_QUICK set)")
    for item in items:
        case = getattr(item, "callspec", None) and item.callspec.params.get("case_name")
        if case is not None and case not in QUICK_CASES:
            item.add_marker(skip)
