"""GMAT golden-data harness (design doc §23.1, REQ-VV-002). See golden.py."""

from .golden import (
    build_time_scales_fixture,
    compare_time_scales,
    offsets_from_rows,
    parse_report,
    regenerate_time_scales_fixture,
    write_time_scales_script,
)

__all__ = [
    "build_time_scales_fixture",
    "compare_time_scales",
    "offsets_from_rows",
    "parse_report",
    "regenerate_time_scales_fixture",
    "write_time_scales_script",
]
