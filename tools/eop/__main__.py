"""Refresh the committed IERS EOP fixture (ground-side; never run in CI).

    PYTHONPATH=tools uv run python -m eop --out tests/golden/finals.all.iau2000.txt

Downloads ``finals.all.iau2000`` (trying the mirrors in order), sanity-checks
that it parses, and writes it **verbatim** to ``--out``. The file is committed as
static data and parsed on the C++ side — see ``tools/eop/finals.py``. Equivalent
to ``curl <url> -o <out>`` but with mirror fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .finals import fetch_finals2000a, parse_finals2000a


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eop", description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help="force a single EOP URL (default: try MIRRORS in order)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output path for the raw product"
    )
    args = parser.parse_args(argv)

    try:
        text = fetch_finals2000a(args.url)
    except OSError as exc:
        print(f"eop: download failed: {exc}", file=sys.stderr)
        return 1

    rows = parse_finals2000a(text)
    if len(rows) < 2:
        print(
            f"eop: downloaded content parsed to only {len(rows)} row(s) — not a finals file?",
            file=sys.stderr,
        )
        return 1

    args.out.write_text(text)
    print(
        f"eop: ok — {len(rows)} rows [{rows[0].mjd_utc:.0f}, {rows[-1].mjd_utc:.0f}] -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
