"""Refresh the committed SGP4 verification fixtures (ground-side; never run in CI).

    PYTHONPATH=tools uv run python -m tle --out tests/golden

Downloads the AIAA 2006-6753 companion package, extracts ``SGP4-VER.TLE`` and
``tforverf.out``, sanity-checks that both parse, and writes them **verbatim**
under ``--out`` with their upstream names — so a refresh is an overwrite and
nothing sits between the published artifact and the committed one. See
``tools/tle/verification.py`` for why this particular set is the acceptance test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .verification import PACKAGE_URL, fetch, write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tle", description=__doc__)
    parser.add_argument("--url", default=PACKAGE_URL, help="package URL")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory to write the fixtures into"
    )
    args = parser.parse_args(argv)

    if not args.out.is_dir():
        print(f"tle: {args.out} is not a directory", file=sys.stderr)
        return 1

    try:
        fetched = fetch(args.url)
    except (OSError, KeyError) as exc:
        print(f"tle: fetch failed: {exc}", file=sys.stderr)
        return 1

    # Cheap structural checks. Not a parse of the algorithm's input — that is the
    # C++ side's job — just enough that a login page or a truncated download
    # cannot be committed as a fixture.
    by_name = {f.filename: f.data for f in fetched}
    tle = by_name["SGP4-VER.TLE"].decode("ascii", errors="replace")
    cases = sum(1 for line in tle.splitlines() if line.startswith("1 "))
    if cases < 30:
        print(
            f"tle: SGP4-VER.TLE parsed to only {cases} case(s) — truncated?",
            file=sys.stderr,
        )
        return 1
    expected = by_name["tforverf.out"].decode("ascii", errors="replace")
    rows = sum(1 for line in expected.splitlines() if line.strip().endswith("xx"))
    if rows < 30:
        print(
            f"tle: tforverf.out has only {rows} satellite block(s) — truncated?",
            file=sys.stderr,
        )
        return 1

    written = write(args.out, fetched)
    print(
        f"tle: ok — {cases} verification cases, {rows} expected-output blocks -> "
        f"{', '.join(str(p) for p in written)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
