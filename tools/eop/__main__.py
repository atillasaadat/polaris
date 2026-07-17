"""Regenerate the committed IERS EOP fixture (ground-side; never run in CI).

    PYTHONPATH=tools uv run python -m eop \
        --start-mjd 58849 --end-mjd 61041 \
        --out tests/golden/eop.json

Downloads IERS ``finals2000A.all`` (or reads ``--input`` for an offline copy),
parses the Bulletin A columns, trims to the MJD window, and writes the JSON
fixture the C++ ``EopTable`` consumes. The fixture is committed data — see
``tools/eop/finals.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .finals import (
    MIRRORS,
    build_fixture,
    fetch_finals2000a,
    parse_finals2000a,
    trim,
    write_fixture,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eop", description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help="force a single EOP URL (default: try MIRRORS in order)",
    )
    parser.add_argument(
        "--input", type=Path, help="local finals2000A.all (skips download)"
    )
    parser.add_argument(
        "--start-mjd", type=float, required=True, help="inclusive start MJD (UTC)"
    )
    parser.add_argument(
        "--end-mjd", type=float, required=True, help="inclusive end MJD (UTC)"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output JSON fixture path"
    )
    args = parser.parse_args(argv)

    try:
        text = args.input.read_text() if args.input else fetch_finals2000a(args.url)
    except OSError as exc:
        print(f"eop: cannot read EOP source: {exc}", file=sys.stderr)
        return 1

    rows = trim(parse_finals2000a(text), args.start_mjd, args.end_mjd)
    if len(rows) < 2:
        print(
            f"eop: only {len(rows)} row(s) in [{args.start_mjd}, {args.end_mjd}]; "
            "EopTable needs >=2 to interpolate",
            file=sys.stderr,
        )
        return 1

    # Provenance records where the data *originates*: an explicit --url, else the
    # canonical IERS endpoint — even when read from a local --input copy.
    source = args.url or MIRRORS[0]
    write_fixture(args.out, build_fixture(rows, source))
    print(
        f"eop: ok — {len(rows)} rows [{rows[0].mjd_utc:.0f}, {rows[-1].mjd_utc:.0f}] -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
