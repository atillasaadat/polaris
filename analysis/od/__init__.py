"""Orbit-determination Monte Carlo analysis (design doc §8.3, §9.2, §13, §23.2).

Reads the per-cycle records written by the C++ campaign driver
(``tests/mc/orbit_od_mc.cpp``, target ``polaris_orbit_od_mc``) and produces the
statistics, figures and report the §8.3 filter's own unit tests cannot: over a
week of real dynamics, against a truth stack the filter does not have, does the
estimate stay bounded, does the covariance still mean what it says, and what
happens when the receiver misbehaves in each of the ways §9.2 says it can.

The split is the standing one in ``analysis/CLAUDE.md``: the runs are C++ — the
filter under test *is* ``gnc::OrbitOd``, driven against the real receiver model,
so no GNC math is reimplemented here — and the sampling statistics, plotting and
report generation are Python, which is one of that rule's named exceptions.

What is a criterion here and what is a measurement is a deliberate line: no
requirement writes a number on the filter's accuracy, so the accuracy figures
are reported and the criteria are the claims with a threshold this campaign did
not choose — the chi-square consistency intervals, the NIS gate's own configured
rejection rate, and the fault policy. See :mod:`analysis.od.report`.

See ``analysis/od/README.md`` for how to fly the campaign.
"""

from __future__ import annotations

from analysis.od.ensemble import EnsembleAxis, ensemble_covariance
from analysis.od.records import Campaign, ScenarioRun, load_campaign
from analysis.od.report import od_report
from analysis.od.statistics import (
    CampaignStatistics,
    ConsistencyInterval,
    ErrorSummary,
    RegimeSummary,
    ScenarioStatistics,
    consistency_interval,
    summarise,
)

__all__ = [
    "Campaign",
    "CampaignStatistics",
    "ConsistencyInterval",
    "EnsembleAxis",
    "ErrorSummary",
    "RegimeSummary",
    "ScenarioRun",
    "ScenarioStatistics",
    "consistency_interval",
    "ensemble_covariance",
    "load_campaign",
    "od_report",
    "summarise",
]
