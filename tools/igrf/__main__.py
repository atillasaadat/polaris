"""Refresh the committed IGRF-14 coefficients and golden fixture (ground-side).

    PYTHONPATH=tools uv run --group igrf python -m igrf coeffs \\
        --out tests/golden/igrf14coeffs.txt

    PYTHONPATH=tools uv run --group igrf python -m igrf golden \\
        --pyigrf <extracted>/pyIGRF14 --out tests/golden/igrf14_reference.csv

``coeffs`` downloads the IAGA table, sanity-checks that it parses, and writes it
**verbatim** — the file is committed as static data and parsed on the C++ side
(design doc §3.7). ``golden`` regenerates the derived field fixture with the
IAGA reference implementation, which is a fetch input and is never committed;
see ``tools/igrf/reference.py``. Neither runs in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .coeffs import DEFAULT_URL, fetch, parse_igrf_coeffs
from .reference import (
    PYIGRF_URL,
    SAMPLE_POINTS,
    SAMPLE_YEARS,
    cross_check,
    load_igrf_utils,
    synthesize,
    write_fixture,
)


def _refresh_coeffs(args: argparse.Namespace) -> int:
    try:
        text = fetch(args.url)
    except OSError as exc:
        print(f"igrf: download failed: {exc}", file=sys.stderr)
        return 1

    try:
        parsed = parse_igrf_coeffs(text)
    except ValueError as exc:
        print(f"igrf: downloaded content did not parse: {exc}", file=sys.stderr)
        return 1

    args.out.write_text(text)
    print(
        f"igrf: ok — {len(parsed.rows)} coefficients to degree {parsed.nmax}, "
        f"epochs {parsed.epochs[0]:.1f}..{parsed.epochs[-1]:.1f} -> {args.out}"
    )
    return 0


def _regenerate_golden(args: argparse.Namespace) -> int:
    try:
        igrf_utils = load_igrf_utils(args.pyigrf)
    except FileNotFoundError as exc:
        print(f"igrf: {exc}", file=sys.stderr)
        return 2

    shc = args.shc or args.pyigrf / "SHC_files" / "IGRF14.SHC"
    if not shc.is_file():
        print(f"igrf: {shc} not found (see {PYIGRF_URL})", file=sys.stderr)
        return 2

    model = igrf_utils.load_shcfile(str(shc))
    committed = parse_igrf_coeffs(args.coeffs.read_text())
    try:
        cross_check(model, committed)
    except ValueError as exc:
        print(f"igrf: {shc} disagrees with {args.coeffs}: {exc}", file=sys.stderr)
        return 1

    rows = synthesize(igrf_utils, model)
    write_fixture(
        args.out,
        rows,
        coeffs_name=args.coeffs.name,
        coeffs_sha256=hashlib.sha256(args.coeffs.read_bytes()).hexdigest(),
        shc_name=shc.name,
        shc_sha256=hashlib.sha256(shc.read_bytes()).hexdigest(),
        nmax=int(model.parameters["nmax"]),
    )
    print(
        f"igrf: ok — {len(rows)} rows "
        f"({len(SAMPLE_POINTS)} points x {len(SAMPLE_YEARS)} epochs) -> {args.out}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="igrf", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    coeffs = subparsers.add_parser(
        "coeffs", help="download the IAGA coefficient table verbatim"
    )
    coeffs.add_argument("--url", default=DEFAULT_URL, help="coefficient source URL")
    coeffs.add_argument(
        "--out", type=Path, required=True, help="output path for the raw product"
    )
    coeffs.set_defaults(handler=_refresh_coeffs)

    golden = subparsers.add_parser(
        "golden", help="regenerate the derived field fixture with pyIGRF14"
    )
    golden.add_argument(
        "--pyigrf",
        type=Path,
        required=True,
        help=f"extracted pyIGRF14 directory (download: {PYIGRF_URL})",
    )
    golden.add_argument(
        "--shc",
        type=Path,
        default=None,
        help="SHC file (default: <pyigrf>/SHC_files/IGRF14.SHC)",
    )
    golden.add_argument(
        "--coeffs",
        type=Path,
        default=Path("tests/golden/igrf14coeffs.txt"),
        help="committed IAGA table to cross-check against",
    )
    golden.add_argument("--out", type=Path, required=True, help="fixture output path")
    golden.set_defaults(handler=_regenerate_golden)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
