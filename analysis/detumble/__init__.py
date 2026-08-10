"""B-dot detumble Monte Carlo analysis (design doc §13, §23.2; REQ-ACTL-001).

Reads the per-run records written by the C++ campaign driver
(``tests/mc/detumble_mc.cpp``, target ``polaris_detumble_mc``) and produces the
statistics, figures and report the requirement's recorded "owed" item asks for:
the distribution of the time to the ``DetumbleExitRadps`` completion predicate,
and the Safe-mode handover time a percentile of it supports.

The split is the standing one in ``analysis/CLAUDE.md``: the runs are C++ —
they fly the real ``ClosedLoop`` against the real forked deployment, so no GNC
math is reimplemented here — and the sampling statistics, plotting and report
generation are Python, which is one of that rule's named exceptions.

See ``analysis/detumble/README.md`` for how to run the campaign.
"""

from __future__ import annotations

from analysis.detumble.records import RunRecord, load_records
from analysis.detumble.report import detumble_report
from analysis.detumble.statistics import (
    DetumbleStatistics,
    summarise,
    wilks_sample_size,
)

__all__ = [
    "DetumbleStatistics",
    "RunRecord",
    "detumble_report",
    "load_records",
    "summarise",
    "wilks_sample_size",
]
