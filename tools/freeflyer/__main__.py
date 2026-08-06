"""Command-line front door: ``python -m freeflyer <subcommand>``.

Subcommands
-----------
``status``
    Report every discovered FreeFlyer installation and its license state.
``viz --stream <file> [--follow] [--pace N] [--headless]``
    Render a Polaris truth-state stream (``POLARIS_SIM_STREAM`` JSONL) in
    interactive FreeFlyer windows — live with ``--follow`` while a sim runs,
    or as a replay of a finished run. ``--pace 10`` replays 10× faster than
    real time; ``--pace 0`` renders as fast as the display draws.

Run with ``PYTHONPATH=tools`` from the repository root (the pytest config
does the same), or via ``uv run python -m freeflyer …``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import locate, viz


def _cmd_status(_args: argparse.Namespace) -> int:
    installs = locate.find_installs()
    if not installs:
        print("no FreeFlyer installation found (POLARIS_FF_DIR to point at one)")
        return 1
    for inst in installs:
        state = inst.license_info.get("license", "UNLICENSED")
        expiry = inst.license_info.get("expires", "")
        run = "runnable" if inst.runnable else "not runnable from this Python"
        print(f"[{inst.platform}] {inst.install_dir}")
        print(f"    license: {state} {expiry}   ({run})")
    return 0


def _cmd_viz(args: argparse.Namespace) -> int:
    install = locate.find_runnable_licensed()
    if install is None:
        print("no runnable licensed FreeFlyer found", file=sys.stderr)
        return 1
    stream = Path(args.stream)
    states = viz.follow(stream, max_fps=args.fps) if args.follow else viz.replay(stream)
    pace = None if args.follow or args.pace == 0 else args.pace
    try:
        frames = viz.run_viz(
            install, states, pace=pace, windowed=not args.headless, max_fps=args.fps
        )
    except KeyboardInterrupt:
        print("\ninterrupted — engine killed", file=sys.stderr)
        return 130
    print(f"rendered {frames} frames from {stream}")
    return 0 if frames else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m freeflyer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="report discovered installations and licenses")

    p_viz = sub.add_parser(
        "viz", help="render a truth-state stream in FreeFlyer windows"
    )
    p_viz.add_argument(
        "--stream", required=True, help="JSONL file POLARIS_SIM_STREAM wrote"
    )
    p_viz.add_argument(
        "--follow", action="store_true", help="tail a live run instead of replaying"
    )
    p_viz.add_argument(
        "--pace", type=float, default=1.0, help="replay rate (1 = real time, 0 = max)"
    )
    p_viz.add_argument(
        "--headless", action="store_true", help="no windows (smoke testing)"
    )
    p_viz.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="render-rate ceiling (default 2; the WSLg software renderer "
        "sustains little more — 0 disables)",
    )

    args = parser.parse_args()
    return {"status": _cmd_status, "viz": _cmd_viz}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
