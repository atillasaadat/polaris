"""Refresh the committed CelesTrak space-weather fixture (ground-side; not CI).

    PYTHONPATH=tools uv run python -m spaceweather --out tests/golden/SW-All.csv

Downloads ``SW-All.csv``, sanity-checks that it parses, and writes it
**verbatim** to ``--out``. The file is committed as static data and parsed on the
C++ side — see ``sim/world/space_weather_file.cpp``. Equivalent to
``curl <url> -o <out>`` but with a parse gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .spaceweather import fetch_sw_all, parse_sw_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spaceweather", description=__doc__)
    parser.add_argument("--url", default=None, help="override the CelesTrak URL")
    parser.add_argument(
        "--out", type=Path, required=True, help="output path for the raw CSV"
    )
    args = parser.parse_args(argv)

    try:
        text = fetch_sw_all(args.url)
    except OSError as exc:
        print(f"spaceweather: download failed: {exc}", file=sys.stderr)
        return 1

    rows = parse_sw_all(text)
    if len(rows) < 2:
        print(
            f"spaceweather: parsed only {len(rows)} row(s) — not an SW-All file?",
            file=sys.stderr,
        )
        return 1

    args.out.write_text(text)
    print(
        f"spaceweather: ok — {len(rows)} daily rows [{rows[0].date}, {rows[-1].date}] -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
