"""Shared fixtures for the analysis suites (design doc §13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.control import load_vehicle

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed reference vehicle. Every quantitative assertion in this suite
#: is made against *this* file, never against a transcription of it.
REFERENCE_CONFIG = REPO_ROOT / "config" / "spacecraft" / "leo_smallsat.yaml"


@pytest.fixture(scope="session")
def reference_config() -> Path:
    """Path to the committed reference spacecraft config."""
    return REFERENCE_CONFIG


@pytest.fixture(scope="session")
def vehicle():
    """The as-flown reference LEO smallsat, loaded from the committed config."""
    return load_vehicle(REFERENCE_CONFIG)


def od_sample(index: int, **overrides) -> dict:
    """One synthetic orbit-OD sample row, in the driver's own schema.

    Defaults describe a healthy cycle: solution published, fix delivered and
    folded in, errors well inside the claimed sigma. Every test here builds its
    case by overriding the few fields it is actually about, so a row that looks
    unremarkable in a test is unremarkable in the campaign too.
    """
    row = {
        "kind": "sample",
        "run": 0,
        "scenario": "nominal",
        "t_s": float(index) * 10.0,
        "regime": "nominal",
        "pos_err_m": 0.5,
        "vel_err_mps": 0.005,
        "pos_sigma_m": 0.7,
        "vel_sigma_mps": 0.009,
        "err_ric_m": [0.3, 0.3, 0.2],
        "sigma_ric_m": [0.4, 0.4, 0.4],
        "nees": 6.0,
        "nis": 3.0,
        "solution_valid": 1,
        "fix_valid": 1,
        "fix_accepted": 1,
        "age_s": 0.0,
        "rejected_total": 0,
    }
    row.update(overrides)
    return row


def write_od_shard(
    path: Path,
    groups: list[tuple[int, str, list[dict]]],
    *,
    truncate_last: bool = False,
) -> Path:
    """Write a JSONL shard: one meta line per (run, scenario) then its samples.

    Parameters
    ----------
    groups : list of (run, scenario, rows)
        Each entry becomes one meta line followed by its sample rows, with the
        run and scenario stamped onto every row so a caller cannot build a shard
        the driver would never emit.
    truncate_last : bool
        Drop the final newline *and* half the final line, reproducing the shard
        a still-flying campaign is read from.
    """
    lines: list[str] = []
    for run, scenario, rows in groups:
        meta = {
            "kind": "meta",
            "run": run,
            "scenario": scenario,
            "intent": f"synthetic {scenario}",
            "cycle_period_s": 10.0,
            "fix_latency_s": 0.0,
            "duration_s": 10.0 * len(rows),
        }
        lines.append(json.dumps(meta))
        for row in rows:
            lines.append(json.dumps({**row, "run": run, "scenario": scenario}))

    text = "\n".join(lines)
    if truncate_last:
        text = text[: -len(lines[-1]) // 2]
    else:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path
