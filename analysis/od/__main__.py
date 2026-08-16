"""Command-line entry point: the orbit-determination campaign gate.

``python -m analysis.od <records>`` reads the JSONL the C++ campaign driver
wrote, reduces it to statistics, writes an **interactive HTML report** to
``<out>/index.html``, opens it in a browser, and **exits non-zero if any
criterion fails**.

``<records>`` is a shard, a directory of ``*.jsonl`` shards, or several of
either — a campaign is normally flown sharded across cores and reassembled here
rather than by concatenating files.

The HTML page is the default output for the same reason the sizing report's is:
the claim is a shape over time — an error staying inside a covariance envelope
through a week of outages and spoofs — and no table shows that. ``--no-browser``
writes without opening, which is what CI and the tests use, and
``POLARIS_NO_BROWSER`` in the environment does the same for every report CLI at
once. ``--print`` additionally writes the plain-text report to stdout; that
rendering is the record the tests assert on and is always written to
``<out>/od_report.txt`` either way.

The exit status is the point. This is a verification gate, not a viewer: a
filter whose covariance has stopped meaning what it says, or whose plausibility
band has stopped catching implausible fixes, should fail a command rather than
require someone to notice a number on a page.

Usage
-----
.. code-block:: bash

   # Fly a campaign shard (C++; built in the unsanitized tree -- see the README)
   ./build-fprime-automatic-native/bin/Linux/polaris_orbit_od_mc \\
       --first-run 0 --runs 4 --duration-s 86400 \\
       --out build-artifacts/orbit-od-mc/shard-0.jsonl

   # Read every shard and produce the verdict
   PYTHONPATH=tools uv run --group analysis python -m analysis.od \\
       build-artifacts/orbit-od-mc

``PYTHONPATH=tools`` is not optional: like every package here this one lives
beside tooling that is not an installed distribution. ``pytest.ini`` arranges it
for the test suite.

References
----------
Design doc §8.3, §9.2, §13, §23.2; ``analysis/od/README.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.common.report_html import open_in_browser
from analysis.od.html import write_html
from analysis.od.records import load_campaign
from analysis.od.report import od_report
from analysis.od.statistics import summarise

#: Where artifacts land when ``--out`` is not given. Derived output, never
#: committed.
DEFAULT_OUTPUT_DIR = Path("build-artifacts/orbit-od")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.od",
        description="Orbit-determination Monte Carlo campaign report and gate.",
    )
    parser.add_argument(
        "records",
        nargs="+",
        help="JSONL shard(s) or a directory of them, as written by polaris_orbit_od_mc",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output directory (default {DEFAULT_OUTPUT_DIR})",
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
        help="also write the plain-text report to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the campaign analysis and return the process exit status.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 when every criterion passes, 1 otherwise, 2 on a usage or data error.
        A data error is deliberately distinct from a failing criterion: "the
        campaign says the filter is inconsistent" and "there was no campaign"
        are different answers and must not share an exit code.
    """
    args = _parser().parse_args(argv)
    out_dir = args.out or DEFAULT_OUTPUT_DIR

    try:
        campaign = load_campaign([Path(p) for p in args.records])
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not campaign.runs:
        print("error: the records contain no completed runs", file=sys.stderr)
        return 2

    stats = summarise(campaign)
    source = ", ".join(str(p) for p in campaign.paths)
    report = od_report(stats, source, campaign.truncated)

    out_dir.mkdir(parents=True, exist_ok=True)
    text = report.format_text()
    (out_dir / "od_report.txt").write_text(text, encoding="utf-8")

    page = write_html(campaign, stats, report, out_dir)
    print(f"orbit-OD report: {page}")
    open_in_browser(page, enabled=not args.no_browser)

    if args.print_report:
        print()
        print(text)
        print()

    return 0 if report.passes else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
