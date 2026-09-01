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
``run --scenario <gtest filter> [--replay] [--stream <file>]``
    Run a SITL scenario **and** watch it, in one command and one shell: starts
    the integration binary with ``POLARIS_SIM_STREAM`` pointed at a fresh
    stream, then follows that stream in the windows. ``--replay`` simulates
    first and replays the finished run at ``--pace``, which is what anything
    faster than real time wants — a full-orbit row simulates 94 minutes in 12
    seconds, and following that live coalesces the orbit into a couple of dozen
    frames. The sim runs in WSL and
    the rendering hosts on Windows, the same split ``viz`` uses — this
    subcommand just owns both ends of it and cleans the sim up on the way out.

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
import subprocess
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


#: Where the SITL rows live, relative to the repository root.
_SITL_BINARY = Path(
    "build-fprime-automatic-native-ut/bin/Linux/polaris_integration_tests"
)


def _repo_root() -> Path:
    """The checkout root — this file is ``<root>/tools/freeflyer/__main__.py``."""
    return Path(__file__).resolve().parents[2]


def _cmd_run(args: argparse.Namespace) -> int:
    """Start a SITL scenario and follow it in one shell.

    The two halves run where they must: the simulation is a Linux binary and
    stays in WSL, the rendering hosts on Windows (``winhost``). This command is
    the only place that knows both, which is why it is not itself relaunched —
    a Windows child could not start the Linux binary.
    """
    binary = Path(args.binary) if args.binary else _repo_root() / _SITL_BINARY
    if not binary.exists():
        print(
            f"{binary} not found — build the integration tests first:\n"
            "  uv run cmake --build build-fprime-automatic-native-ut "
            "--target polaris_integration_tests -j4",
            file=sys.stderr,
        )
        return 1

    stream = (
        Path(args.stream)
        if args.stream
        else Path(f"/tmp/polaris-viz-{os.getpid()}.jsonl")
    )
    # The sim opens this path, it does not build the tree; a missing directory
    # is otherwise a viewer that waits forever on a file nobody can write.
    stream.parent.mkdir(parents=True, exist_ok=True)
    stream.unlink(missing_ok=True)  # never follow a previous run's states

    log = stream.with_suffix(".log")
    env = dict(os.environ, POLARIS_SIM_STREAM=str(stream))
    print(
        f"[run] {args.scenario}\n[run] stream {stream}\n[run] sim log {log}", flush=True
    )
    with log.open("w") as sink:
        sim = subprocess.Popen(
            [str(binary), f"--gtest_filter={args.scenario}"],
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
        try:
            if args.replay:
                # Watch it afterwards, at a pace you choose. This is the right
                # mode for anything that outruns the renderer, which is most
                # short scenarios: a full-orbit row simulates 94 minutes in 12
                # seconds of wall clock (460x real time), and *following* that
                # live coalesces the whole orbit into a couple of dozen frames,
                # because the follower always skips to the newest state rather
                # than fall behind. Replay draws the whole arc instead.
                print(
                    "[run] simulating (the windows open when it finishes)...",
                    flush=True,
                )
                sim.wait()
                viz_argv = [
                    "viz",
                    "--stream",
                    str(stream),
                    "--pace",
                    str(args.pace),
                    "--fps",
                    str(args.fps),
                    "--view",
                    args.view,
                ]
            else:
                viz_argv = [
                    "viz",
                    "--stream",
                    str(stream),
                    "--follow",
                    "--fps",
                    str(args.fps),
                    "--view",
                    args.view,
                ]
            if winhost.should_relaunch():
                code = winhost.relaunch(viz_argv)
            else:
                install = locate.find_runnable_licensed()
                if install is None:
                    print("no runnable licensed FreeFlyer found", file=sys.stderr)
                    return 1
                states = (
                    viz.replay(stream)
                    if args.replay
                    else viz.follow(stream, max_fps=args.fps)
                )
                code = (
                    0
                    if viz.run_viz(
                        install,
                        states,
                        pace=(args.pace if args.replay else None),
                        max_fps=args.fps,
                        view=args.view,
                    )
                    else 1
                )
        except KeyboardInterrupt:
            code = 130
        finally:
            # The viewer returns when the stream goes quiet, which is usually
            # the run having finished — but a Ctrl-C or a dead engine gets here
            # too, and a SITL binary left running holds ports and a PrmDb.
            if sim.poll() is None:
                sim.terminate()
                try:
                    sim.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    sim.kill()
    if sim.returncode not in (0, -15, 143):
        print(
            f"[run] the scenario exited {sim.returncode}; tail of {log}:",
            file=sys.stderr,
        )
        for line in log.read_text(errors="replace").splitlines()[-15:]:
            print("   ", line, file=sys.stderr)
        return sim.returncode or 1
    return code


def _cmd_sgp4_fixture(_args) -> int:
    """Regenerate the committed FreeFlyer SGP4/TLE cross-validation fixture."""
    from freeflyer import sgp4_fixture

    return sgp4_fixture.main()


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

    p_run = sub.add_parser("run", help="run a SITL scenario and watch it, in one shell")
    p_run.add_argument(
        "--scenario",
        required=True,
        help="gtest filter, e.g. SitlAttitudeControl.DetumblesThenAcquiresSunPointing "
        "(--gtest_list_tests on the integration binary lists them all)",
    )
    p_run.add_argument(
        "--stream", help="where the run writes its states (default: a fresh /tmp file)"
    )
    p_run.add_argument(
        "--binary", help="integration-test binary (default: the build tree's)"
    )
    p_run.add_argument(
        "--replay",
        action="store_true",
        help="simulate first, then replay the whole run — right for anything "
        "that outruns the renderer (a full-orbit row simulates 94 min in 12 s, "
        "which following live coalesces to a few dozen frames)",
    )
    p_run.add_argument(
        "--pace",
        type=float,
        default=100.0,
        help="replay rate with --replay (default 100x real time; 0 = as fast as it draws)",
    )
    p_run.add_argument(
        "--fps", type=float, default=12.0, help="render-rate ceiling (default 12)"
    )
    p_run.add_argument(
        "--view",
        choices=("orbit", "close", "both"),
        default="both",
        help="which window(s)",
    )

    sub.add_parser(
        "sgp4-fixture",
        help="regenerate tests/golden/freeflyer_sgp4.json from a live FreeFlyer",
    )

    args = parser.parse_args()
    return {
        "status": _cmd_status,
        "viz": _cmd_viz,
        "panel": _cmd_panel,
        "run": _cmd_run,
        "sgp4-fixture": _cmd_sgp4_fixture,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
