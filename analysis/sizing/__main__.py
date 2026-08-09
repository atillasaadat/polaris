"""Command-line entry point: the ADCS actuator-sizing gate.

``python -m analysis.sizing config/spacecraft/leo_smallsat.yaml`` loads the
committed vehicle, sizes its wheels and rods against the §5.3 disturbance
environment and the sizing drivers, derives the flight tuning the design
implies, writes the figures and both renderings of the report, **opens the HTML
one in a browser**, and **exits non-zero if any criterion fails**.

The HTML page (``<out>/index.html``, :mod:`analysis.sizing.html`) is the default
output because the headline result — the momentum envelope with the certified
ceiling nested inside the hardware one — is a 3D object a reader has to rotate.
``--no-browser`` writes it without opening anything, which is what CI and the
tests use; ``--print`` additionally writes the plain-text report to stdout, which
is unchanged in content and remains the record the tests assert on. The text
rendering is always written to ``<out>/sizing_report.txt`` either way.

That exit status is the point of the module. Sizing is a design gate: a vehicle
whose actuators cannot hold the momentum its environment produces, or whose
detumble exit threshold sits below what its magnetometer can resolve, should be
caught before a simulation spends an hour demonstrating it. Warnings — including
the standing one that no requirement is written on actuator sizing — qualify the
report and do **not** fail it, per the convention in ``analysis/CLAUDE.md``.

Examples
--------
Run the gate on the reference vehicle and open the report::

    PYTHONPATH=tools uv run --group analysis python -m analysis.sizing \\
        config/spacecraft/leo_smallsat.yaml

The same, headless, with the console report as well::

    PYTHONPATH=tools uv run --group analysis python -m analysis.sizing \\
        config/spacecraft/leo_smallsat.yaml --no-browser --print

Size the same vehicle for a harsher tip-off and twice-per-orbit unloading::

    uv run --group analysis python -m analysis.sizing \\
        config/spacecraft/leo_smallsat.yaml --tipoff-deg-s 10 --desat-orbits 0.5

Notes
-----
``tools/`` must be importable so the config compiler's loader can be used
(``PYTHONPATH=tools``); ``pytest.ini`` arranges that for the test suite.
"""

from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
import webbrowser
from pathlib import Path

import numpy as np

from analysis.control.vehicle import load_vehicle
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.html import write_html
from analysis.sizing.plots import DEFAULT_OUTPUT_DIR, write_all, write_text_report
from analysis.sizing.report import (
    format_budget,
    format_derived,
    sizing_analysis,
    sizing_report,
)

#: Matplotlib figures embedded into the HTML page, by filename. The two the
#: interactive figures do not replace; see the comment at their use below.
STATIC_FIGURES = ("momentum_drivers.png", "magnetorquer_sizing.png")


def _is_wsl() -> bool:
    """True on Windows Subsystem for Linux.

    WSL reports a Microsoft kernel in ``/proc/version``; the environment
    variable is only set for interactive shells, so the kernel string is the
    reliable test.
    """
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _open_in_browser(page: Path) -> None:
    """Open *page*, or say plainly how to open it, without ever failing the run.

    :mod:`webbrowser` assumes a browser inside the machine it runs on. A WSL
    distro usually has none: the browser is on the Windows side, so the module
    falls through to ``gio``/``xdg-open`` and those report "no application for
    text/html". The report has already been written at that point, so a failure
    to *display* it must not look like a failure to *produce* it — the analysis
    exit code belongs to the criteria, not to the desktop.

    On WSL the page is handed to the Windows shell (``wslview`` if the wslu
    package is installed, otherwise ``explorer.exe`` on the translated path),
    which opens the user's real browser. Everywhere else :mod:`webbrowser` is
    correct and is used unchanged.
    """
    target = str(page.resolve())
    if not _is_wsl():
        if not webbrowser.open(page.resolve().as_uri()):
            print(f"could not open a browser; the report is at {target}")
        return

    if shutil.which("wslview"):
        if subprocess.run(["wslview", target], check=False).returncode == 0:
            return
    windows_path = ""
    if shutil.which("wslpath"):
        translated = subprocess.run(
            ["wslpath", "-w", target], capture_output=True, text=True, check=False
        )
        if translated.returncode == 0:
            windows_path = translated.stdout.strip()
    if windows_path and shutil.which("explorer.exe"):
        # explorer.exe exits 1 even on success, so its status says nothing;
        # what matters is whether the call could be made at all.
        try:
            subprocess.run(["explorer.exe", windows_path], check=False)
            return
        except OSError:
            pass
    print(
        f"no browser reachable from WSL — open this from Windows:\n  {windows_path or target}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the sizing analysis and return the process exit status.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments, excluding the program name. ``None`` reads
        :data:`sys.argv`.

    Returns
    -------
    int
        ``0`` when every criterion passes, ``1`` when any fails.
    """
    parser = argparse.ArgumentParser(
        prog="python -m analysis.sizing",
        description=(
            "ADCS actuator sizing and design validation for a committed "
            "spacecraft configuration (design doc SS7, SS8.5): actuator "
            "envelopes, the disturbance-torque budget, wheel and magnetorquer "
            "sizing with 30%% margin, and the flight tuning the design implies. "
            "Exits non-zero on any FAIL."
        ),
    )
    parser.add_argument(
        "config", type=Path, help="path to a config/spacecraft/*.yaml file"
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=None,
        help="hardware catalog directory (default: config/hardware beside the "
        "config, else this repository's config/hardware)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the figures and the rendered report "
        "(default: build-artifacts/analysis/sizing)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip the matplotlib figures; the HTML page and the text report "
        "are still written, the page without the static figures it would "
        "otherwise embed",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="write the HTML report without opening a browser (CI and tests)",
    )
    parser.add_argument(
        "--print",
        dest="print_report",
        action="store_true",
        help="also print the plain-text report, the disturbance budget and the "
        "derived-parameter justifications to stdout",
    )
    parser.add_argument(
        "--tipoff-deg-s",
        type=float,
        default=None,
        help="separation tip-off rate [deg/s] (default: 5, REQ-ACTL-001's value)",
    )
    parser.add_argument(
        "--desat-orbits",
        type=float,
        default=None,
        help="desaturation interval in orbits (default: 1)",
    )
    parser.add_argument(
        "--slew-deg-s",
        type=float,
        default=None,
        help="commanded slew rate [deg/s]; without it, slew agility is reported "
        "parametrically rather than judged",
    )
    args = parser.parse_args(argv)

    vehicle = load_vehicle(args.config, args.hardware)
    overrides: dict[str, float] = {}
    if args.tipoff_deg_s is not None:
        overrides["tipoff_rate_radps"] = float(np.deg2rad(args.tipoff_deg_s))
    if args.desat_orbits is not None:
        overrides["desat_interval_s"] = args.desat_orbits * vehicle.orbit.period_s
    if args.slew_deg_s is not None:
        overrides["slew_rate_radps"] = float(np.deg2rad(args.slew_deg_s))
    assumptions = SizingAssumptions(**overrides)

    written = (
        [] if args.no_plots else write_all(vehicle, args.out, args.config, assumptions)
    )
    analysis = sizing_analysis(vehicle, assumptions)
    report = sizing_report(vehicle, args.config, assumptions, analysis)
    if args.no_plots:
        # The text rendering is the record, not a figure: skipping the plots
        # must not skip it. write_all() already wrote it on the other branch.
        write_text_report(analysis, args.out, args.config)

    # Only the two matplotlib figures with no interactive counterpart are
    # embedded: the log-axis driver comparison (a log scale is the only way the
    # drivers and the envelope share one plot) and the magnetorquer authority
    # pair. The 3D envelope and the disturbance bars are superseded by their
    # rotatable/hoverable versions above.
    static = [p for p in written if p.name in STATIC_FIGURES]
    page = write_html(
        analysis, report, args.out or DEFAULT_OUTPUT_DIR, static_figures=static
    )
    print(f"sizing report: {page}")
    if not args.no_browser:
        _open_in_browser(page)

    if args.print_report:
        print()
        print(report.format_text())
        print()
        print(format_budget(analysis))
        print()
        print(format_derived(analysis))
    return 0 if report.passes else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
