"""Shared fixtures for the linear control-analysis suite (design doc §13)."""

from __future__ import annotations

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
