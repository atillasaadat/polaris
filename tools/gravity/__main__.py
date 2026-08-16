"""Ground-side EGM2008 tooling (never run in CI).

Two subcommands, both derivations *from* the upstream ICGEM `.gfc` (design doc
§3.7) rather than substitutes for it:

``truncate`` — refresh the committed `.gfc` reference datum::

    PYTHONPATH=tools uv run python -m gravity truncate \\
        --max-degree 200 --out tests/golden/EGM2008_to200.gfc

Downloads the full EGM2008 `.gfc` (or reads a local copy via --input) and
truncates it to --max-degree in the native format. The full model is ~100 MB; the
committed degree-200 window is ~1.5 MB. See tools/gravity/gfc.py.

``cxx-header`` — regenerate the flight-side low-degree coefficient table::

    PYTHONPATH=tools uv run python -m gravity cxx-header \\
        --input tests/golden/EGM2008_to200.gfc \\
        --max-degree 8 --out lib/gnc/egm2008_low_degree.hpp

De-normalizes and emits the committed C++ header the onboard orbit filter's force
model carries (§8.3). See tools/gravity/cxxtable.py.

For backwards compatibility, invoking with no subcommand behaves as ``truncate``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cxxtable import MAX_SAFE_DEGREE, emit_header
from .gfc import DEFAULT_URL, count_coeffs, fetch, truncate


def _source_text(args: argparse.Namespace) -> str:
    return args.input.read_text() if args.input else fetch(args.url)


def _truncate(args: argparse.Namespace) -> int:
    trimmed = truncate(_source_text(args), args.max_degree)
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


def _cxx_header(args: argparse.Namespace) -> int:
    if args.input is None:
        print(
            "gravity: cxx-header needs --input, the committed .gfc it derives from",
            file=sys.stderr,
        )
        return 1
    text = emit_header(args.input.read_text(), args.max_degree, str(args.input))
    args.out.write_text(text)
    print(f"gravity: ok — degree {args.max_degree} table -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gravity", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    for name, default_degree, help_text in (
        ("truncate", 200, "write a degree-truncated .gfc in the native format"),
        ("cxx-header", 8, "write the flight-side unnormalized C++ coefficient table"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--url", default=DEFAULT_URL, help="ICGEM EGM2008 .gfc URL")
        p.add_argument("--input", type=Path, help="local .gfc (skips download)")
        p.add_argument(
            "--max-degree",
            type=int,
            default=default_degree,
            help=f"truncation degree/order (cxx-header caps at {MAX_SAFE_DEGREE})",
        )
        p.add_argument("--out", type=Path, required=True, help="output path")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("pick a subcommand: truncate | cxx-header")

    try:
        return _truncate(args) if args.command == "truncate" else _cxx_header(args)
    except (OSError, ValueError) as exc:
        print(f"gravity: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
