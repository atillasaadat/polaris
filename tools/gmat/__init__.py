"""GMAT golden-data harness (design doc §23.1, REQ-VV-002).

See golden.py (time scales) and propagation.py (orbit propagation).
"""

from .golden import (
    build_time_scales_fixture,
    compare_time_scales,
    offsets_from_rows,
    parse_report,
    regenerate_time_scales_fixture,
    write_time_scales_script,
)
from .propagation import (
    build_propagation_fixture,
    compare_propagation,
    regenerate_propagation_fixture,
    samples_from_rows,
    write_propagation_script,
)

__all__ = [
    "build_propagation_fixture",
    "build_time_scales_fixture",
    "compare_propagation",
    "compare_time_scales",
    "offsets_from_rows",
    "parse_report",
    "regenerate_propagation_fixture",
    "regenerate_time_scales_fixture",
    "samples_from_rows",
    "write_propagation_script",
    "write_time_scales_script",
]
