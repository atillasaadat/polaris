"""Refresh the committed EGM2008 `.gfc` fixture (ground-side; never run in CI).

    PYTHONPATH=tools uv run python -m gravity \
        --max-degree 200 --out tests/golden/EGM2008_to200.gfc

Downloads the full EGM2008 `.gfc` (or reads a local copy via --input), truncates
to --max-degree in the native format, and writes it. The full model is ~100 MB;
the committed degree-200 window is ~1.5 MB. See tools/gravity/gfc.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gfc import DEFAULT_URL, count_coeffs, fetch, truncate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gravity", description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="ICGEM EGM2008 .gfc URL")
    parser.add_argument(
        "--input", type=Path, help="local full/partial .gfc (skips download)"
    )
    parser.add_argument(
        "--max-degree", type=int, default=200, help="truncation degree/order"
    )
    parser.add_argument("--out", type=Path, required=True, help="output .gfc path")
    args = parser.parse_args(argv)

    try:
        text = args.input.read_text() if args.input else fetch(args.url)
    except OSError as exc:
        print(f"gravity: cannot read EGM2008 source: {exc}", file=sys.stderr)
        return 1

    trimmed = truncate(text, args.max_degree)
    n = count_coeffs(trimmed)
    if n < 3:
        print(
            f"gravity: only {n} coefficient line(s) after truncation — not a .gfc?",
            file=sys.stderr,
        )
        return 1

    args.out.write_text(trimmed)
    print(f"gravity: ok — {n} coeffs to degree {args.max_degree} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
