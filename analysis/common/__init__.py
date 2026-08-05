"""Shared building blocks for every Polaris analysis tool (design doc §13).

The convention this package exists to enforce is in ``analysis/CLAUDE.md``:
**every analysis with a pass/fail criterion produces a report and plots that
show the verdict on their face.** The report is structured first
(:class:`analysis.common.report.AnalysisReport` — what tests assert on) and
rendered second (a plain-text table written beside the figures — what a human
reads); the plots carry the requirement thresholds as annotated lines with the
measured value and verdict in text, so a figure is legible in grayscale and
without the surrounding prose.

``analysis.control`` is the first consumer. The momentum-budget, detumble-MC,
contact-scheduling and link-budget tools reuse these rather than reinventing
them.
"""
