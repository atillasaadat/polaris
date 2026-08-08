"""Command-line entry point: reduce a detumble campaign to its verdict.

``python -m analysis.detumble build-artifacts/detumble-mc/runs.jsonl`` reads the
records the C++ driver wrote, computes the distribution of time-to-completion and
the tolerance bound it supports, writes the figures and the rendered report,
prints the report, and **exits non-zero if any criterion fails**.

What the exit status means here is narrower than for
:mod:`analysis.control`, and deliberately so. The control gate fails a
*configuration*; this one fails a *campaign*: the fast-phase criteria are
REQ-ACTL-001's own bound re-measured across the dispersion, and the rest check
that the campaign is large enough and uncensored enough for the bound it quotes.
The proposed Safe-mode handover time is **not** a criterion — no requirement is
written on it yet, and inventing a threshold here to pass against would be the
"threshold tuned to the measurement" defect. It is reported, plotted and
argued; whether it becomes REQ-ACTL-001's time bound is a requirements decision.

Examples
--------
Summarise a finished campaign, writing artifacts under the default location::

    uv run --group analysis python -m analysis.detumble \\
        build-artifacts/detumble-mc/runs.jsonl

Notes
-----
See ``analysis/detumble/README.md`` for how to fly the campaign that produces
the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.detumble.plots import write_all
from analysis.detumble.records import load_records
from analysis.detumble.report import detumble_report
from analysis.detumble.statistics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HANDOVER_MARGIN,
    DEFAULT_QUANTILE,
    summarise,
)


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
        prog="python -m analysis.detumble",
        description=(
            "B-dot detumble Monte Carlo analysis (design doc SS13, SS23.2): the "
            "residual-spin tail distribution, the tolerance bound on its 95th "
            "percentile, and the Safe-mode handover time it supports. Exits "
            "non-zero on any FAIL."
        ),
    )
    parser.add_argument(
        "records",
        type=Path,
        help="campaign JSONL written by polaris_detumble_mc, or a directory of shards",
    )
    parser.add_argument(
        "--orbit-period-s",
        type=float,
        default=5677.0,
        help="orbit period of the flown scenario [s], for expressing times in orbits",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_QUANTILE,
        help="quantile the tolerance bound covers",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="confidence in the tolerance bound",
    )
    parser.add_argument(
        "--handover-margin",
        type=float,
        default=DEFAULT_HANDOVER_MARGIN,
        help="fractional margin carried over the bound into the proposed handover time",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the figures and the rendered report "
        "(default: build-artifacts/analysis/detumble)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip figure generation; still prints the report and gates on it",
    )
    args = parser.parse_args(argv)

    records = load_records(args.records)
    stats = summarise(
        records,
        orbit_period_s=args.orbit_period_s,
        quantile=args.quantile,
        confidence=args.confidence,
        handover_margin=args.handover_margin,
    )
    report = detumble_report(stats, str(args.records))
    text = report.format_text()
    if not args.no_plots:
        write_all(records, stats, text, args.out, passed=report.passes)
    print(text)
    return 0 if report.passes else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
