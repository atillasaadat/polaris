"""Regenerate the committed solar-system-body Chebyshev ephemeris fixture (ground-side).

    PYTHONPATH=tools uv run --group ephem python -m ephem \\
        --kernel de440s.bsp --out tests/golden/de440_bodies.cheb

Downloads `de440s.bsp` if it is not already present, fits geocentric Sun and Moon
Chebyshev segments over the requested window, and writes the plain-text fixture
that `sim/world/ephemeris_file.*` parses. The kernel itself is **never
committed** — it is a fetch input, like the full EGM2008 `.gfc` (design doc §3.7).

The default fit parameters sit at the accuracy floor of the geocentric
representation: shortening the Sun interval below 8 days or raising its degree
past 12 does not improve the residual, because what is left is the difference
between DE440's own segment structures rather than error in our fit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .de440 import DEFAULT_URL, _MOON, _SUN, PLANETS, fetch, fit_body
from .writer import write_fixture

# 2026-01-01T00:00:00 TDB through 2027-01-01T00:00:00 TDB.
_DEFAULT_JD_START = 2_461_041.5
_DEFAULT_JD_END = 2_461_406.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ephem", description=__doc__)
    parser.add_argument(
        "--kernel",
        type=Path,
        default=Path("de440s.bsp"),
        help="SPK kernel path; downloaded from NAIF if absent",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="kernel source URL")
    parser.add_argument("--out", type=Path, required=True, help="fixture output path")
    parser.add_argument("--jd-start", type=float, default=_DEFAULT_JD_START)
    parser.add_argument("--jd-end", type=float, default=_DEFAULT_JD_END)
    parser.add_argument("--sun-interval-days", type=float, default=8.0)
    parser.add_argument("--sun-degree", type=int, default=12)
    parser.add_argument("--moon-interval-days", type=float, default=4.0)
    parser.add_argument("--moon-degree", type=int, default=12)
    # Geocentric planet motion carries the same monthly EMB wobble as the Sun's
    # (~4700 km) on top of a slow heliocentric drift, so the Sun's cadence works;
    # 16 days halves the segment count at the same residual class. At planetary
    # distances a km-level residual is < 1e-8 relative — far below the point-mass
    # model error itself.
    parser.add_argument("--planet-interval-days", type=float, default=16.0)
    parser.add_argument("--planet-degree", type=int, default=12)
    args = parser.parse_args(argv)

    try:
        from jplephem.spk import SPK
    except ImportError:
        print("jplephem is required: uv run --group ephem ...", file=sys.stderr)
        return 2

    if not args.kernel.exists():
        print(f"fetching {args.url} -> {args.kernel}", file=sys.stderr)
        fetch(args.url, args.kernel)

    digest = hashlib.sha256(args.kernel.read_bytes()).hexdigest()
    kernel = SPK.open(str(args.kernel))

    sun, sun_residual = fit_body(
        kernel,
        _SUN,
        args.jd_start,
        args.jd_end,
        args.sun_interval_days,
        args.sun_degree,
    )
    moon, moon_residual = fit_body(
        kernel,
        _MOON,
        args.jd_start,
        args.jd_end,
        args.moon_interval_days,
        args.moon_degree,
    )

    bodies = {"sun": sun, "moon": moon}
    residuals = {"sun": sun_residual, "moon": moon_residual}
    for name, code in PLANETS.items():
        segments, residual = fit_body(
            kernel,
            code,
            args.jd_start,
            args.jd_end,
            args.planet_interval_days,
            args.planet_degree,
        )
        bodies[name] = segments
        residuals[name] = residual

    write_fixture(
        args.out,
        source_url=args.url,
        kernel_name=args.kernel.name,
        kernel_sha256=digest,
        jd_start=args.jd_start,
        jd_end=args.jd_end,
        bodies=bodies,
        residuals=residuals,
    )

    summary = ", ".join(
        f"{name} {len(segs)} segments (max residual {residuals[name]:.3g} m)"
        for name, segs in bodies.items()
    )
    print(f"wrote {args.out}: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
