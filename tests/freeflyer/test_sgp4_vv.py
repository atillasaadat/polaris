"""FreeFlyer SGP4/TLE drift check against the committed fixture (REQ-ODP-003).

The Polaris-vs-FreeFlyer comparison itself lives in C++
(``tests/golden/sgp4_external_golden_test.cpp``), where it belongs: it needs the
propagator and ``frames::eciFromTeme``, and it must run on every PR, where
FreeFlyer is not installed. That test reads the committed fixture
``tests/golden/freeflyer_sgp4.json``.

This file guards the *other* end of that arrangement — that the committed
fixture still equals what a live FreeFlyer produces. Without it the fixture is a
snapshot nobody ever re-derives, and a FreeFlyer upgrade that moved the numbers
would go unnoticed while the C++ test stayed green against stale data. Same
role the weekly ``golden`` lane plays for GMAT.

Skips visibly without a licensed install (``conftest.py`` owns the skip), so a
machine without the seat is never silently green. Runs as a ``pre-push`` hook.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "golden" / "freeflyer_sgp4.json"

#: This compares FreeFlyer against *itself*, so the band is tight on purpose:
#: anything beyond fixture round-off means the engine, its reference data or the
#: generated script changed. It is not an accuracy band and must not be widened
#: to accommodate one — regenerate the fixture and review the diff instead.
DRIFT_TOL_KM = 1.0e-6


@pytest.fixture(scope="module")
def committed_fixture() -> dict:
    assert _FIXTURE.exists(), f"missing committed fixture {_FIXTURE}"
    return json.loads(_FIXTURE.read_text())


@pytest.mark.verifies("REQ-ODP-003")
def test_committed_fixture_agrees_with_live_freeflyer(ff_install, committed_fixture):
    """Every committed sample still reproduces on this FreeFlyer install."""
    from freeflyer import vv

    drift: list[str] = []
    checked = 0
    for case in committed_fixture["cases"]:
        times = [s["t_s"] for s in case["samples"]]
        fresh = vv.run_tle_case(
            ff_install, case["name"], case["tle_line1"], case["tle_line2"], times
        )
        assert len(fresh) == len(
            case["samples"]
        ), f"{case['name']}: FreeFlyer returned {len(fresh)} of {len(times)} samples"
        for old, new in zip(case["samples"], fresh):
            assert math.isclose(
                old["t_s"], new["t_s"], abs_tol=1e-6
            ), f"{case['name']}: sample times drifted apart"
            d = math.dist(old["position_km"], new["position_km"])
            if d > DRIFT_TOL_KM:
                drift.append(
                    f"{case['name']} t={old['t_s']}s: {d:.3e} km "
                    f"(band {DRIFT_TOL_KM:.1e} km)"
                )
            checked += 1

    assert checked > 0, "no samples compared — the fixture is empty"
    assert not drift, (
        "live FreeFlyer disagrees with tests/golden/freeflyer_sgp4.json:\n"
        + "\n".join(drift)
        + "\n\nRegenerate with `uv run --group analysis python -m freeflyer.sgp4_fixture` "
        "and review the diff; do not widen the band."
    )


@pytest.mark.verifies("REQ-ODP-003")
def test_fixture_covers_every_orbital_regime(committed_fixture):
    """The regime coverage is asserted, not just assumed from the case names.

    A fixture regenerated from a shortened case list would still pass the drift
    check above — it would simply verify fewer things and report success. This
    is the guard against that: the regimes that make the comparison meaningful
    (near-Earth and deep space, resonant and not, across a wide range of radii)
    must all still be present.
    """
    names = {c["name"] for c in committed_fixture["cases"]}
    required = {
        "leo_near_earth",
        "leo_sso",
        "molniya_12h_resonant",
        "geo_24h_synchronous",
        "deep_space_non_resonant",
        "deep_space_decaying",
    }
    assert (
        required <= names
    ), f"fixture lost regime coverage: missing {sorted(required - names)}"

    radii = []
    for case in committed_fixture["cases"]:
        for s in case["samples"]:
            radii.append(math.dist([0.0, 0.0, 0.0], s["position_km"]))
    # The angular-vs-linear argument the C++ bands rest on needs a wide spread of
    # radii to mean anything; 7000 km to beyond 42 000 km is what it was derived
    # against.
    assert (
        min(radii) < 8000.0
    ), f"fixture no longer reaches LEO radii (min {min(radii):.0f} km)"
    assert (
        max(radii) > 40000.0
    ), f"fixture no longer reaches synchronous radii (max {max(radii):.0f} km)"
