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

    **Run it from WSL as always**: when a licensed Windows install is present
    the command re-enters itself under Windows Python so the engine renders on
    the GPU, translating the stream path to its ``\\wsl.localhost`` form. The
    sim, the stream file and this command line all stay where they were; only
    the renderer crosses (``winhost.py``). Windowed output on Linux is refused
    rather than silently software-rendered.
``panel --stream <file> [--port N] [--host H]``
    Same replay, but seekable: serves a browser transport control
    (play/pause, seek slider, jump-to-start/timestamp, pace) on
    ``http://127.0.0.1:8765`` and drives the FreeFlyer windows from it.
    The page is a single self-contained document, so a Grafana dashboard
    can embed it in an iframe panel. Runs until Ctrl-C.

    Hosted on Windows like ``viz``, which is also where the panel listens:
    open the printed URL in a **Windows** browser. WSL has its own network
    namespace, so that URL from inside WSL will not reach it.

Run with ``PYTHONPATH=tools`` from the repository root (the pytest config
does the same), or via ``uv run python -m freeflyer …``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import locate, panel, viz, winhost


def _cmd_status(_args: argparse.Namespace) -> int:
    installs = locate.find_installs()
    if not installs:
        print("no FreeFlyer installation found (POLARIS_FF_DIR to point at one)")
        return 1
    host = locate.find_render_host()
    for inst in installs:
        state = inst.license_info.get("license", "UNLICENSED")
        expiry = inst.license_info.get("expires", "")
        run = "runnable" if inst.runnable else "not runnable from this Python"
        roles = []
        if inst.runnable and inst.licensed:
            roles.append("V&V")
        if host is not None and inst.install_dir == host.install_dir:
            roles.append("renders viz")
        if inst.sdk_dir is None:
            roles.append("no Runtime API SDK")
        print(f"[{inst.platform}] {inst.install_dir}")
        print(f"    license: {state} {expiry}   ({run})")
        if roles:
            print(f"    role: {', '.join(roles)}")
    if host is not None and host.platform == "windows":
        interpreter = winhost.find_windows_python()
        print(
            f"\nvisualization hosts on Windows via "
            f"{interpreter or 'NO WINDOWS PYTHON FOUND (set POLARIS_WIN_PYTHON)'}"
        )
    return 0


def _cmd_viz(args: argparse.Namespace) -> int:
    if not args.headless and winhost.should_relaunch():
        return winhost.relaunch(sys.argv[1:])
    install = locate.find_runnable_licensed()
    if install is None:
        print("no runnable licensed FreeFlyer found", file=sys.stderr)
        return 1
    stream = Path(args.stream)
    states = viz.follow(stream, max_fps=args.fps) if args.follow else viz.replay(stream)
    pace = None if args.follow or args.pace == 0 else args.pace
    try:
        frames = viz.run_viz(
            install,
            states,
            pace=pace,
            windowed=not args.headless,
            max_fps=args.fps,
            view=args.view,
        )
    except KeyboardInterrupt:
        print("\ninterrupted — engine killed", file=sys.stderr)
        return 130
    print(f"rendered {frames} frames from {stream}")
    return 0 if frames else 1


def _cmd_panel(args: argparse.Namespace) -> int:
    if not args.headless and winhost.should_relaunch():
        return winhost.relaunch(sys.argv[1:])
    install = locate.find_runnable_licensed()
    if install is None:
        print("no runnable licensed FreeFlyer found", file=sys.stderr)
        return 1
    stream = Path(args.stream)
    states = list(viz.replay(stream))
    if not states:
        print(f"{stream}: no states to replay", file=sys.stderr)
        return 1
    playback = panel.Playback(panel.stream_times(states), pace=args.pace)
    server = panel.serve(playback, host=args.host, port=args.port)
    where = " (open in a Windows browser)" if os.name == "nt" else ""
    print(
        f"control panel: http://{args.host}:{server.server_address[1]}/"
        f"{where}  (Ctrl-C to quit)"
    )
    try:
        viz.run_viz_panel(
            install,
            states,
            playback,
            windowed=not args.headless,
            max_fps=args.fps,
            view=args.view,
        )
    except KeyboardInterrupt:
        print("\ninterrupted — engine killed", file=sys.stderr)
        return 130
    finally:
        server.shutdown()
    return 0


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
        default=12.0,
        help="render-rate ceiling (default 12; the Windows GPU host sustains "
        "~17 fps on this scene — 0 disables)",
    )
    p_viz.add_argument(
        "--view",
        choices=("orbit", "close", "both"),
        default="both",
        help="which window(s) to render (both cost the same on the GPU host)",
    )

    p_panel = sub.add_parser(
        "panel", help="seekable replay driven by a browser control panel"
    )
    p_panel.add_argument(
        "--stream", required=True, help="JSONL file POLARIS_SIM_STREAM wrote"
    )
    p_panel.add_argument(
        "--host", default="127.0.0.1", help="panel bind address (default localhost)"
    )
    p_panel.add_argument(
        "--port", type=int, default=8765, help="panel port (default 8765; 0 = any free)"
    )
    p_panel.add_argument(
        "--pace", type=float, default=1.0, help="initial playback rate (default 1)"
    )
    p_panel.add_argument(
        "--headless", action="store_true", help="no windows (smoke testing)"
    )
    p_panel.add_argument(
        "--fps", type=float, default=12.0, help="render-rate ceiling (default 12)"
    )
    p_panel.add_argument(
        "--view",
        choices=("orbit", "close", "both"),
        default="both",
        help="which window(s) to render (both cost the same on the GPU host)",
    )

    args = parser.parse_args()
    return {"status": _cmd_status, "viz": _cmd_viz, "panel": _cmd_panel}[args.command](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main())
