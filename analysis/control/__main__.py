"""Command-line entry point: the pre-simulation control-design gate.

``python -m analysis.control config/spacecraft/leo_smallsat.yaml`` loads the
committed vehicle, runs the full linear analysis
(:func:`analysis.control.report.control_analysis_report`), writes the figures and
the rendered report (:func:`analysis.control.plots.write_all`), prints the report,
and **exits non-zero if any criterion fails**.

That exit status is the point of the module. The analysis is a design gate: a
configuration whose margins, controllability or observability do not meet their
requirements should stop a run before the simulation spends an hour confirming
it. Warnings — including the SISO-validity warning the reference vehicle raises
at full wheel momentum — qualify the report and do **not** fail it, per the
standing convention in ``analysis/CLAUDE.md``.

Examples
--------
Run the gate on the reference vehicle, writing artifacts under the default
``build-artifacts`` location::

    PYTHONPATH=tools uv run --group analysis python -m analysis.control \\
        config/spacecraft/leo_smallsat.yaml

Notes
-----
``tools/`` must be importable so the config compiler's loader can be used
(``PYTHONPATH=tools``); ``pytest.ini`` arranges that for the test suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.control.plots import write_all
from analysis.control.report import control_analysis_report
from analysis.control.vehicle import load_vehicle


def main(argv: list[str] | None = None) -> int:
    """Run the analysis and return the process exit status.

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
        prog="python -m analysis.control",
        description=(
            "Linear control analysis of a committed spacecraft configuration "
            "(design doc SS8.5): stability margins, controllability, "
            "observability. Exits non-zero on any FAIL."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="path to a config/spacecraft/*.yaml file",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=None,
        help="hardware catalog directory (default: config/hardware beside the config)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the figures and the rendered report "
        "(default: build-artifacts/analysis/control)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip figure generation; still prints the report and gates on it",
    )
    args = parser.parse_args(argv)

    vehicle = load_vehicle(args.config, args.hardware)
    if args.no_plots:
        report = control_analysis_report(vehicle, args.config)
    else:
        write_all(vehicle, args.out, args.config)
        report = control_analysis_report(vehicle, args.config)
    print(report.format_text())
    return 0 if report.passes else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
