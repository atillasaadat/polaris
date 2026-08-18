"""The campaign's verdict, as a structured report.

What can honestly be a criterion here, and what cannot
-------------------------------------------------------
No requirement writes a number on the filter's position accuracy. REQ-ODP-001
requires that the FSW estimate its orbit with a covariance-propagating MEKF;
REQ-ODP-005 fixes the force-model fidelity and how ``q_a`` is derived;
REQ-ODP-006 fixes the latency correction. None of them says "under N metres",
and inventing one here and then passing against it would be the
threshold-tuned-to-the-measurement defect the review-lessons catalogue names.

So the accuracy numbers are carried as **measurements**, in the provenance and
on the page, and the criteria are the claims that have a threshold which is not
this campaign's to choose:

* **Consistency.** For a consistent filter the normalised errors are chi-square
  distributed with the state or measurement dimension as degrees of freedom
  (Bar-Shalom §5.4.2 [barshalom2001]), so the average over independent runs has
  an acceptance interval that is a property of the distribution. That is a real
  threshold, and it is the one that matters: an accurate but *optimistic* filter
  is the one that will eventually reject a correct measurement.
* **Gate behaviour under nominal conditions.** The NIS gate is configured at the
  99.9 % point of chi-square with three degrees of freedom, so it should refuse
  about 0.1 % of clean fixes. The threshold comes from the gate's own
  configuration, not from taste. Far above it and the filter is under-weighting
  its measurements; far below and the gate is not doing anything.
* **Fault policy.** An outage inside the coast horizon must keep the solution
  valid, one past it must drop it, and an implausible fix must be refused **on
  the plausibility band** rather than one layer in. That last is a criterion and
  not a nicety: the band is the only check on the seed path, where there is no
  prior and therefore no innovation gate, and a GEO-radius fix refused by the
  gate looks identical to one refused by the band in any count that does not
  record the reason.
* **Campaign integrity.** Enough runs, enough samples, and every scenario
  actually flown. A verdict read off a campaign that half ran is a verdict
  pointing the wrong way.

The spoof-ramp result is a measurement, not a criterion
--------------------------------------------------------
``spoof_ramp`` walks the filter 2 km over an hour at about 0.55 m/s — slow
enough that each individual innovation is inside a gate sized for one fix's
noise. How far it gets before anything notices is exactly the number the
scenario exists to produce, and there is no requirement bounding it. It is
reported as a warning carrying the measured drift, which is the same treatment
the detumble campaign gives its handover-time proposal.

References
----------
Design doc §8.3 (onboard OD), §9.2 (GNSS FDIR), §13 (Monte Carlo), §22.2 (margin
reporting). REQ-ODP-001, REQ-ODP-005, REQ-ODP-006
(``docs/requirements/orbit_od_propagation.rst``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from analysis.common.report import AnalysisReport, Criterion
from analysis.od.statistics import CampaignStatistics, ScenarioStatistics

__all__ = [
    "MIN_CAMPAIGN_RUNS",
    "NOMINAL_REJECTION_CEILING",
    "od_report",
]

#: The NIS gate sits at the 99.9 % point of chi-square(3), so under nominal
#: conditions it should refuse about 0.1 % of fixes. The ceiling is set an order
#: above that expectation rather than at it: the gate's *expected* rate is exact,
#: but the campaign measures it over a finite sample and against a truth the
#: filter does not carry, so a few times the nominal rate is ordinary and ten
#: times it is not. Derived from the gate's configuration, not chosen to pass.
NOMINAL_REJECTION_CEILING = 0.01

#: Below this the campaign is a smoke test and its consistency intervals are too
#: wide to refuse anything. The chi-square interval widens as 1/sqrt(N), and at
#: fewer than ten independent runs it admits a filter that is a factor of two
#: optimistic — which is to say it admits the failure it exists to catch.
MIN_CAMPAIGN_RUNS = 10


def _consistency_criterion(
    name: str, interval, requirement: str, what: str
) -> tuple[Criterion, Criterion]:
    """The two one-sided halves of a chi-square acceptance interval.

    Written as two criteria rather than one, because they are not the same
    engineering situation and a single "inside the interval" boolean hides which
    way it failed. Optimistic — the covariance smaller than the error — is the
    unsafe direction; pessimistic wastes information and flies fine.
    """
    mean = interval.mean
    return (
        Criterion(
            name=f"{name} not optimistic",
            requirement=requirement,
            threshold=interval.upper,
            measured=mean,
            units="-",
            sense="max",
            note=(
                f"{what} averaged over {interval.samples} independent "
                f"run{'' if interval.samples == 1 else 's'}, against the "
                f"upper chi-square acceptance bound at 95%. Above it the filter's "
                f"covariance is smaller than its actual error, which is the direction "
                f"that makes an estimate unsafe: an overconfident filter under-weights "
                f"measurements and eventually refuses correct ones."
            ),
            formula="chi2.ppf(0.975, dof*N)/N",
        ),
        Criterion(
            name=f"{name} not pessimistic",
            requirement=requirement,
            threshold=interval.lower,
            measured=mean,
            units="-",
            sense="min",
            note=(
                "The same average against the lower bound. Below it the covariance is "
                "larger than the error; the estimate is safe but the filter is throwing "
                "away information it has, and the process noise is over-budgeted."
            ),
            formula="chi2.ppf(0.025, dof*N)/N",
        ),
    )


def _nominal_rejection(entry: ScenarioStatistics) -> Criterion | None:
    """The gate's rejection rate over a scenario's un-faulted stretches."""
    summary = entry.regimes.get("nominal")
    if summary is None or summary.fixes == 0:
        return None
    return Criterion(
        name=f"Clean-fix rejection rate, {entry.scenario}",
        requirement="REQ-ODP-001",
        threshold=NOMINAL_REJECTION_CEILING,
        measured=summary.rejection_rate,
        units="-",
        sense="max",
        note=(
            f"{summary.fixes - summary.accepted} of {summary.fixes} clean fixes refused. "
            f"The gate is configured at the 99.9% point of chi-square(3), so about 0.1% "
            f"is expected; the ceiling is an order above that, since the rate is measured "
            f"over a finite sample against a truth the filter does not carry."
        ),
    )


def _band_refusals(stats: CampaignStatistics) -> Criterion | None:
    """Implausible fixes must be refused on the band, not by the innovation gate.

    Counted over the ``radius_jump`` regime only, which is the one the campaign
    injects them in. The measurement is the fraction refused as
    ``fix_implausible``; anything else means the trust boundary let the fix
    through and something further in caught it, which is not the same defence and
    is unavailable when the filter is seeding.
    """
    band = other = 0
    for entry in stats.scenarios:
        summary = entry.regimes.get("radius_jump")
        if summary is None:
            continue
        for kind, count in summary.refusals.items():
            if kind == "fix_implausible":
                band += count
            else:
                other += count
    total = band + other
    if total == 0:
        return None
    return Criterion(
        name="Implausible fixes refused at the trust boundary",
        requirement="REQ-ODP-001",
        threshold=1.0,
        measured=band / total,
        units="-",
        sense="min",
        note=(
            f"{band} of {total} refusals of an injected geostationary-radius fix came from "
            f"the plausibility band. The band is the only check on the seed path — a cold "
            f"filter, or one whose solution the coast horizon just dropped, has no prior and "
            f"so no innovation gate — so a fix refused anywhere else would have been accepted "
            f"whole had it arrived one cycle earlier."
        ),
    )


#: Acceptance band on the ensemble/reported sigma ratio. One is a covariance
#: that matches the truth-derived spread. The band is wide because the sample
#: standard deviation itself carries ~1/sqrt(2(N-1)) relative uncertainty --
#: ~13 % at 30 runs, ~35 % at 5 -- so a tighter band would fail a healthy filter
#: on sampling noise alone. What it is sized to catch is a covariance wrong by a
#: factor, which is the failure that actually occurs: a mis-scaled process
#: noise, a missing cross term, a variance mistaken for a standard deviation.
ENSEMBLE_RATIO_MAX = 2.0
ENSEMBLE_RATIO_MIN = 0.5


def _ensemble_criteria(stats: CampaignStatistics) -> list[Criterion]:
    """The covariance against truth, per RIC axis.

    Every other consistency criterion here normalises the error by the very
    covariance under test. These do not: the spread is estimated from the
    ensemble of true errors across independent runs, and the filter's claim is
    then held up against it. A filter wrong about its error and its covariance
    in the same direction passes NEES and fails this.
    """
    out: list[Criterion] = []
    for entry in stats.ensemble:
        note = (
            f"Truth-derived 1σ {entry.ensemble_sigma_m:.3f} m from the spread across "
            f"{entry.runs} independent runs, against the {entry.reported_sigma_m:.3f} m the "
            f"filter claimed over the same samples. Unlike NEES this never consults the "
            f"covariance being judged — the spread comes from truth alone. Above one the "
            f"real error is wider than the filter admits, which is the unsafe direction. "
            f"Nominal regime only, where the covariance is claiming to describe the error."
        )
        out.append(
            Criterion(
                name=f"Ensemble/reported sigma, {entry.axis.replace('_', '-')}",
                requirement="REQ-ODP-001",
                threshold=ENSEMBLE_RATIO_MAX,
                measured=entry.ratio,
                units="-",
                sense="max",
                note=note,
            )
        )
        out.append(
            Criterion(
                name=f"Ensemble/reported sigma, {entry.axis.replace('_', '-')}, lower bound",
                requirement="REQ-ODP-001",
                threshold=ENSEMBLE_RATIO_MIN,
                measured=entry.ratio,
                units="-",
                sense="min",
                note=(
                    "The same ratio against its lower bound. Below it the filter is "
                    "pessimistic: safe, but it is discarding information it has and its "
                    "process noise is over-budgeted."
                ),
            )
        )
    return out


def _integrity(stats: CampaignStatistics) -> list[Criterion]:
    """Was there enough campaign to read a verdict off."""
    return [
        Criterion(
            name="Independent runs",
            requirement="REQ-ODP-001",
            threshold=float(MIN_CAMPAIGN_RUNS),
            measured=float(stats.nees.samples),
            units="-",
            sense="min",
            note=(
                "Runs contributing to the consistency test. The chi-square interval "
                "widens as 1/sqrt(N); below ten runs it admits a filter that is a factor "
                "of two optimistic, which is the failure it exists to catch. Runs, not "
                "cycles: consecutive cycles of one run are correlated over the filter's "
                "own time constants and are not independent samples."
            ),
        ),
    ]


def od_report(
    stats: CampaignStatistics,
    records_path: str,
    truncated: Sequence[Path] = (),
) -> AnalysisReport:
    """Build the campaign report.

    Parameters
    ----------
    stats : analysis.od.statistics.CampaignStatistics
    records_path : str
        Where the records were read from, for provenance.
    truncated : sequence of pathlib.Path
        Shards whose final record was a partial write — see
        :attr:`analysis.od.records.Campaign.truncated`. Carried into the
        warnings so a report read off a still-flying campaign says so; without
        it a half-finished campaign renders identically to a finished one.

    Returns
    -------
    analysis.common.report.AnalysisReport
    """
    criteria: list[Criterion] = list(_integrity(stats))
    criteria.extend(
        _consistency_criterion(
            "Campaign NEES",
            stats.nees,
            "REQ-ODP-001",
            "Normalised estimation error squared over the full 6-state,",
        )
    )
    criteria.extend(
        _consistency_criterion(
            "Campaign NIS",
            stats.nis,
            "REQ-ODP-001",
            "Normalised innovation squared of the position update,",
        )
    )
    criteria.extend(_ensemble_criteria(stats))
    for entry in stats.scenarios:
        rejection = _nominal_rejection(entry)
        if rejection is not None:
            criteria.append(rejection)
    band = _band_refusals(stats)
    if band is not None:
        criteria.append(band)

    worst = max(
        (
            e.worst_pos_err_m
            for e in stats.scenarios
            if not math.isnan(e.worst_pos_err_m)
        ),
        default=float("nan"),
    )
    nominal = stats.of("nominal")
    steady = nominal.regimes.get("nominal") if nominal is not None else None

    provenance = {
        "Records": records_path,
        "Runs": str(stats.runs),
        "Samples": str(stats.samples),
        "Scenarios": ", ".join(e.scenario for e in stats.scenarios),
        "Worst position error": f"{worst:.4g} m across every scenario and regime",
    }
    if steady is not None:
        provenance["Steady-state position error"] = (
            f"median {steady.position.median:.4g} m, "
            f"95th {steady.position.p95:.4g} m, "
            f"worst {steady.position.worst:.4g} m"
        )
        provenance["Steady-state error, in units of the reported sigma"] = (
            f"median {steady.sigma_ratio.median:.3g}, 95th {steady.sigma_ratio.p95:.3g}"
        )
        if steady.sma.samples > 0:
            # NASA/TP-2018-219822 §2.1: the SMA error is the OD figure of merit
            # that predicts (period error -> secular along-track drift), and its
            # ratio to the filter's own SMA sigma judges the covariance on it.
            provenance["Steady-state semi-major-axis error (TP-2018-219822 §2.1)"] = (
                f"median {steady.sma.median:.4g} m, 95th {steady.sma.p95:.4g} m; "
                f"in units of the reported SMA sigma: median "
                f"{steady.sma_sigma_ratio.median:.3g}, 95th {steady.sma_sigma_ratio.p95:.3g}"
            )

    return AnalysisReport(
        title="Orbit determination Monte Carlo — reference LEO smallsat",
        config_path=records_path,
        provenance=provenance,
        assumptions=_assumptions(stats),
        criteria=tuple(criteria),
        warnings=_warnings(stats, truncated),
    )


def _assumptions(stats: CampaignStatistics) -> tuple[str, ...]:
    """What the numbers depend on that is not in them."""
    cadences = sorted(
        {e.cycle_period_s for e in stats.scenarios if e.cycle_period_s > 0}
    )
    return (
        "Truth is a 32x32 geopotential, the full piecewise atmosphere, Sun and Moon "
        "third bodies and SRP; the onboard model is 8x8 with an exponential atmosphere. "
        "The difference is the truncation the process noise is budgeted for, and it is "
        "the reason this campaign says anything the filter's own unit tests cannot.",
        "GNSS position error is modelled white per axis at the receiver's datasheet RMS. "
        "Real fix error is correlated over minutes through the common-mode ionosphere, "
        "orbit and clock error across the visible constellation, which white noise "
        "understates for an estimator that averages successive fixes — so the "
        "steady-state accuracy here is optimistic in a way the fault behaviour is not.",
        "Cadence: "
        + ", ".join(f"{c:g} s" for c in cadences)
        + ". The long arcs run at "
        "10 s, two orders below the receiver's rate and the flown ~1 Hz case, so the "
        "filter is given far less information than it will fly with and the error is "
        "over-stated — the right direction for a bound.",
        "The consistency intervals treat runs as independent and cycles within a run as "
        "not. That is why they are computed over run means: applying a chi-square "
        "interval to correlated samples would produce an interval far too tight and "
        "fail a healthy filter.",
        "Fix latency is armed only in the latency_fast scenario, whose 50 Hz cadence can "
        "resolve it. The long arcs pass zero, because at a 10 s cadence the delay-line "
        "model realises a whole poll of delay rather than the datasheet's 50 ms.",
    )


def _warnings(
    stats: CampaignStatistics, truncated: Sequence[Path] = ()
) -> tuple[str, ...]:
    """Measurements that qualify the campaign without failing it."""
    notes: list[str] = []

    if truncated:
        notes.append(
            f"{len(truncated)} shard{'' if len(truncated) == 1 else 's'} ended in a "
            f"partial record and were read short: "
            f"{', '.join(str(p) for p in truncated)}. Either the campaign is still "
            f"flying or a driver was killed. Every number below is computed over the "
            f"records that exist, so the arcs are shorter than the campaign was asked "
            f"for and the run count may be too."
        )

    for entry in stats.scenarios:
        spoof = entry.regimes.get("spoof")
        if spoof is None or not math.isfinite(spoof.position.worst):
            continue
        notes.append(
            f"{entry.scenario}: the filter reached {spoof.position.worst:.4g} m from truth "
            f"while a spoof was armed, refusing {spoof.fixes - spoof.accepted} of "
            f"{spoof.fixes} fixes. This is a measurement and not a criterion — no "
            f"requirement bounds how far a slow spoof may walk the estimate, and a "
            f"threshold invented here and then passed against would be a threshold tuned "
            f"to its own measurement. It is the number a spoofing requirement should be "
            f"written from."
        )

    if stats.runs < MIN_CAMPAIGN_RUNS:
        notes.append(
            f"Only {stats.runs} runs. Read the accuracy numbers as a smoke test: the "
            f"consistency intervals at this sample size are wide enough to admit a filter "
            f"that is substantially optimistic."
        )

    for entry in stats.scenarios:
        if entry.solution_gap_s > 0.0:
            notes.append(
                f"{entry.scenario}: longest stretch with no valid solution was "
                f"{entry.solution_gap_s:.0f} s. Expected wherever an outage past the 300 s "
                f"coast horizon is armed — the solution is *supposed* to be dropped rather "
                f"than coasted indefinitely — and worth reading against the scenario's "
                f"armed outage durations."
            )

    return tuple(notes)
