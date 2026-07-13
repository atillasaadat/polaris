"""CLI for the GMAT golden harness (design doc §23.1, REQ-VV-002).

Two subcommands, both needing the ``GmatConsole`` binary (``--console`` or
``$GMAT_CONSOLE``) — neither runs in the normal test path:

    python -m gmat regenerate  --out tests/golden/time_scales.json
    python -m gmat drift-check --fixture tests/golden/time_scales.json

``regenerate`` writes a fresh fixture from GMAT (at GMAT's ~µs tolerance);
``drift-check`` regenerates in a temp dir and fails (exit 1) if the committed
fixture disagrees with GMAT beyond tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .golden import compare_time_scales, regenerate_time_scales_fixture


def _resolve_console(arg: str | None) -> str:
    console = arg or os.environ.get("GMAT_CONSOLE")
    if not console:
        sys.exit("no GmatConsole: pass --console PATH or set GMAT_CONSOLE")
    return console


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gmat", description=__doc__)
    parser.add_argument("--console", help="path to GmatConsole (else $GMAT_CONSOLE)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    regen = sub.add_parser("regenerate", help="run GMAT and write the fixture")
    regen.add_argument("--out", required=True, help="fixture path to write")

    drift = sub.add_parser("drift-check", help="fail if committed fixture drifts")
    drift.add_argument("--fixture", required=True, help="committed fixture to check")

    args = parser.parse_args(argv)
    console = _resolve_console(args.console)

    with tempfile.TemporaryDirectory() as tmp:
        fixture = regenerate_time_scales_fixture(console, tmp)

    if args.cmd == "regenerate":
        Path(args.out).write_text(json.dumps(fixture, indent=2) + "\n")
        print(f"wrote {args.out}")
        return 0

    committed = json.loads(Path(args.fixture).read_text())
    drift_msgs = compare_time_scales(committed, fixture)
    if drift_msgs:
        print(f"GMAT drift vs {args.fixture}:")
        for msg in drift_msgs:
            print(f"  - {msg}")
        return 1
    print(f"OK: GMAT agrees with {args.fixture} within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
