"""Serialize fitted Chebyshev segments to the committed `.cheb` fixture format.

Plain text, one segment per line, so the fixture diffs and reviews like every
other committed reference file in this repo (design doc §3.7). Chosen over JSON
deliberately: the C++ reader (`sim/world/ephemeris_file.*`) then needs nothing
but `<sstream>`, and the fixture stays greppable.

Format::

    # comment / provenance lines, ignored by the parser
    seg <body> <mid_ns> <radius_seconds> <degree> <cx...> <cy...> <cz...>

`mid_ns` is TDB nanoseconds since 1970-01-01T00:00:00 (an int64, written without
an exponent so it round-trips exactly). Each component contributes
``degree + 1`` coefficients, in that order, in metres. Coefficients are written
with 17 significant digits — enough to round-trip an IEEE-754 double exactly, so
reloading the fixture reproduces the fit bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

from .de440 import Segment


def _format_segment(body: str, seg: Segment) -> str:
    coefficients = " ".join(f"{c:.17g}" for c in (*seg.cx, *seg.cy, *seg.cz))
    return (
        f"seg {body} {seg.mid_ns} {seg.radius_seconds:.17g} {seg.degree} {coefficients}"
    )


def write_fixture(
    path: Path,
    *,
    source_url: str,
    kernel_name: str,
    kernel_sha256: str,
    jd_start: float,
    jd_end: float,
    bodies: dict[str, list[Segment]],
    residuals: dict[str, float],
) -> None:
    """Write the fixture, provenance header first."""
    lines = [
        "# Polaris geocentric solar-system-body Chebyshev ephemeris fit (Sun, Moon, planets).",
        "#",
        "# DERIVED PRODUCT (design doc SS3.7): computed from the JPL DE440 SPK kernel",
        "# below, which is a fetch input and is NOT committed. Regenerate with:",
        "#   PYTHONPATH=tools uv run --group ephem python -m ephem --out <this file>",
        "#",
        f"# source_url:     {source_url}",
        f"# kernel:         {kernel_name}",
        f"# kernel_sha256:  {kernel_sha256}",
        f"# coverage_jd_tdb: {jd_start:.1f} .. {jd_end:.1f}",
        "#",
        "# Positions are GEOCENTRIC (Earth-centred ICRF/ECI), metres. Times are TDB",
        "# nanoseconds since 1970-01-01T00:00:00.",
        "#",
        "# Max fit residual vs DE440:",
    ]
    for body, residual in sorted(residuals.items()):
        lines.append(f"#   {body}: {residual:.4g} m over {len(bodies[body])} segments")
    lines.append("#")
    lines.append(
        "# seg <body> <mid_ns> <radius_seconds> <degree> <cx...> <cy...> <cz...>"
    )

    for body, segments in sorted(bodies.items()):
        for seg in segments:
            lines.append(_format_segment(body, seg))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
