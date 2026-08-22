"""Reducing the campaign to the statistics a verdict can be read from.

The campaign asks four separable questions and they need different statistics.

**Is the estimate bounded?** Over the campaign arc, against a truth stack the filter does
not have, does the position error stay where the filter is useful. This is a
distribution, summarised by its quantiles and its worst case, per regime — the
nominal stretches and the fault stretches are different populations and pooling
them reports neither.

**Does the covariance mean what it says?** This is the question a worst case
cannot answer: a filter can be accurate and overconfident at the same time, and
the overconfident one is the one that will accept a spoof. The test is the
standard NEES/NIS consistency check (Bar-Shalom §5.4.2 [barshalom2001]): for a
consistent filter the normalised errors are :math:`\\chi^2` distributed with the
state (6) or measurement (3) dimension as degrees of freedom, so the *average*
over :math:`N` independent samples falls inside
:math:`[\\chi^2_{dN}(\\alpha/2), \\chi^2_{dN}(1-\\alpha/2)]/N` with probability
:math:`1-\\alpha`. That interval is a property of the chi-square distribution, not
a tuning knob — which is what makes it usable as a criterion here, where no
requirement writes a number on filter accuracy. Both tests normalise the error
by the covariance under test and so cannot see a filter wrong about the two
together; :mod:`analysis.od.ensemble` is the second, truth-derived estimate that
closes that gap, and :func:`summarise` reports it alongside these.

**Do the gates fire, and does the right one fire?** The NIS gate is configured at
the 99.9 % point of :math:`\\chi^2_3`, so under nominal conditions it should
reject about 0.1 % of fixes: far above that and the filter is under-weighting its
own measurements, far below and the gate is not doing anything. Separately, and
learned the hard way, *which* refusal fired matters as much as whether one did —
a GEO-radius fix rejected by the innovation gate looks like success and is not,
because the innovation gate does not exist on the seed path.

**Is the fault policy the documented one?** An outage inside the coast horizon
must keep the solution valid; one past it must drop it. That is policy from the
configuration, so it is checkable without inventing a threshold.

Independence, and why the NEES interval is used carefully
---------------------------------------------------------
The chi-square interval above assumes independent samples. Consecutive cycles of
one run are emphatically not independent — the filter state is correlated over
its own time constants — so applying the interval to a single run's 60480 samples
would produce an interval far too tight and fail a healthy filter. What is
independent is *runs*: each flies its own dispersion draw and its own noise
stream. So the consistency interval is computed over the run-mean NEES with the
number of **runs** as the sample count, and the per-run means are reported
alongside so a single bad run is visible rather than averaged away. This is the
same reason the detumble campaign takes its tolerance bound at the second order
statistic rather than the first.

Units
-----
Metres, metres per second, seconds. NEES and NIS are dimensionless.

References
----------
Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, Wiley, 2001, §5.4.2 [barshalom2001] — the NEES/NIS consistency
tests and their chi-square acceptance intervals.
Design doc §8.3 (onboard OD), §9.2 (GNSS FDIR), §13, §23.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from analysis.od.ensemble import EnsembleAxis, ensemble_covariance
from analysis.od.records import Campaign, ScenarioRun

__all__ = [
    "CampaignStatistics",
    "ConsistencyInterval",
    "ErrorSummary",
    "RegimeSummary",
    "ScenarioStatistics",
    "consistency_interval",
    "summarise",
]

#: Filter state dimension, for the NEES chi-square degrees of freedom.
STATE_DIM = 6

#: Position measurement dimension, for the NIS chi-square degrees of freedom.
MEASUREMENT_DIM = 3

#: Two-sided significance for the consistency intervals. 5 % is the convention
#: the tracking literature reports these tests at; it is a stated confidence
#: level, not a threshold chosen to make a measurement pass.
ALPHA = 0.05


@dataclass(frozen=True)
class ConsistencyInterval:
    """A chi-square acceptance interval on an average normalised error.

    Attributes
    ----------
    mean : float
        The measured average — of NEES over runs, or of NIS over runs.
    lower, upper : float
        The interval the average must fall in for the filter to be consistent at
        :data:`ALPHA`. Below the interval the filter is *pessimistic* (its
        covariance is larger than its error, which wastes information but is
        safe); above it the filter is *optimistic*, which is the dangerous
        direction because an overconfident filter under-weights measurements and
        eventually stops believing correct ones.
    dof : int
        Degrees of freedom per sample.
    samples : int
        Independent samples the interval was computed for — runs, never cycles.
        See the module docstring.
    """

    mean: float
    lower: float
    upper: float
    dof: int
    samples: int

    @property
    def consistent(self) -> bool:
        """The average falls inside the interval."""
        return bool(self.lower <= self.mean <= self.upper)

    @property
    def optimistic(self) -> bool:
        """Above the interval: the covariance is smaller than the error."""
        return bool(self.mean > self.upper)


def consistency_interval(values: np.ndarray, dof: int) -> ConsistencyInterval:
    """The chi-square acceptance interval for a set of per-run normalised errors.

    Parameters
    ----------
    values : numpy.ndarray
        One average normalised error per independent run. NaNs are dropped:
        a run that judged no measurements contributes no information about
        consistency, which is different from contributing a zero.
    dof : int
        Degrees of freedom of a single sample — :data:`STATE_DIM` for NEES,
        :data:`MEASUREMENT_DIM` for NIS.

    Returns
    -------
    ConsistencyInterval
        With ``mean`` NaN and an empty interval when there is nothing to test.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return ConsistencyInterval(float("nan"), float("nan"), float("nan"), dof, 0)
    total_dof = dof * count
    return ConsistencyInterval(
        mean=float(finite.mean()),
        lower=float(stats.chi2.ppf(ALPHA / 2.0, total_dof) / count),
        upper=float(stats.chi2.ppf(1.0 - ALPHA / 2.0, total_dof) / count),
        dof=dof,
        samples=count,
    )


@dataclass(frozen=True)
class ErrorSummary:
    """The distribution of one error quantity over a set of samples.

    Quantiles rather than a mean and a standard deviation: the error is bounded
    below by zero and has a long right tail wherever a fault is armed, so a
    symmetric summary describes a distribution that is not there.
    """

    samples: int
    median: float
    p95: float
    p99: float
    worst: float

    @staticmethod
    def of(values: np.ndarray) -> "ErrorSummary":
        """Summarise, tolerating an empty selection."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            nan = float("nan")
            return ErrorSummary(0, nan, nan, nan, nan)
        return ErrorSummary(
            samples=int(finite.size),
            median=float(np.median(finite)),
            p95=float(np.percentile(finite, 95.0)),
            p99=float(np.percentile(finite, 99.0)),
            worst=float(finite.max()),
        )


@dataclass(frozen=True)
class RegimeSummary:
    """Error and covariance behaviour while one fault regime was armed.

    Attributes
    ----------
    regime : str
        ``"nominal"``, ``"outage"``, ``"spoof"``, ``"jam"``, ``"clock_jump"``,
        ``"radius_jump"`` or ``"sigma_degrade"``.
    position, velocity : ErrorSummary
        Error distributions [m], [m/s], over samples where the filter published
        a solution. Cycles with no solution are excluded: an absent estimate has
        no error, and counting it as zero would report a dropped solution as
        perfect accuracy.
    sigma_ratio : ErrorSummary
        Distribution of ``pos_err_m / pos_sigma_m`` — how many of its own claimed
        sigmas the filter is actually away from truth. Dimensionless, and the one
        number that reads the same whatever the regime.
    sma : ErrorSummary
        Distribution of ``|sma_err_m|`` [m] — the semi-major-axis error, the OD
        figure of merit NASA/TP-2018-219822 §2.1 recommends (SMA error is period
        error is secular along-track drift). Empty on shards written before the
        MC harness recorded it (Push 73).
    sma_sigma_ratio : ErrorSummary
        ``|sma_err_m| / sma_sigma_m`` — the covariance judged on the metric that
        predicts, not only on the one that fits (TP §2.1.4).
    fixes, accepted, refusals : int, int, dict
        Fixes delivered, fixes folded in, and a count per refusal name. The
        refusal breakdown is the point: *which* layer refused is what
        distinguishes a fault caught at the trust boundary from the same fault
        caught one layer in.
    fine_fixes, fine_accepted : int
        The same two counts over the samples where the filter called its own
        solution ``"fine"``. The gate's false-alarm rate is only meaningful
        there: once the solution is degraded the filter has been coasting past
        ``max_coast_s`` and is refusing fixes because its *state* is stale, not
        because the gate misjudged them (Push 74; REQ-ODP-001). Zero on shards
        written before the driver recorded the quality verdict.
    """

    regime: str
    position: ErrorSummary
    velocity: ErrorSummary
    sigma_ratio: ErrorSummary
    fixes: int
    accepted: int
    refusals: dict[str, int]
    fine_fixes: int = 0
    fine_accepted: int = 0
    sma: ErrorSummary = field(default_factory=lambda: ErrorSummary.of(np.array([])))
    sma_sigma_ratio: ErrorSummary = field(
        default_factory=lambda: ErrorSummary.of(np.array([]))
    )

    @property
    def rejection_rate(self) -> float:
        """Fraction of delivered fixes not folded in; NaN when none arrived."""
        if self.fixes == 0:
            return float("nan")
        return 1.0 - self.accepted / self.fixes

    @property
    def fine_rejection_rate(self) -> float:
        """:attr:`rejection_rate` over the fine-quality samples alone.

        The gate false-alarm rate, and the one a ceiling belongs on. Falls back
        to :attr:`rejection_rate` on a shard that predates the quality verdict,
        so an old campaign still scores rather than silently reporting NaN.
        """
        if self.fine_fixes == 0:
            return self.rejection_rate
        return 1.0 - self.fine_accepted / self.fine_fixes


@dataclass(frozen=True)
class ScenarioStatistics:
    """One scenario, aggregated over every run of it.

    Attributes
    ----------
    scenario, intent : str
        Name and the driver's own rationale for flying it.
    runs : int
        Runs aggregated.
    samples : int
        Total GNC cycles.
    cycle_period_s, fix_latency_s, duration_s : float
        The cadence, latency and arc these runs flew.
    regimes : dict of str to RegimeSummary
        Per-regime behaviour, keyed by regime name and always containing
        ``"nominal"`` when the scenario had any un-faulted stretch.
    nees, nis : ConsistencyInterval
        Filter consistency over the *nominal* stretches only. Deliberately not
        over the fault stretches: while a spoof is armed the filter is *supposed*
        to disagree with a measurement, so a NIS computed there measures the
        fault's size and not the filter's honesty.
    worst_pos_err_m : float
        Worst position error over every regime, for the headline.
    solution_gap_s : float
        Longest continuous stretch with no valid solution [s]. Zero means the
        filter published throughout.
    clean_lockout_s : float
        Longest unbroken stretch of delivered, un-faulted fixes the gate refused
        [s] — how long the filter went on refusing honest fixes, which a rate
        cannot show. Bounded by the degraded horizon: nothing re-seeds sooner
        (§9.2, and the "no N rejections → reseed" rule in ``gnc/orbit_od.hpp``).
    degraded_horizon_s : float
        The degraded coast horizon these runs flew [s], carried from the driver
        so :attr:`clean_lockout_s` is judged against the filter's own bound
        rather than a number copied into the analysis.
    """

    scenario: str
    intent: str
    runs: int
    samples: int
    cycle_period_s: float
    fix_latency_s: float
    duration_s: float
    regimes: dict[str, RegimeSummary]
    nees: ConsistencyInterval
    nis: ConsistencyInterval
    worst_pos_err_m: float
    solution_gap_s: float
    clean_lockout_s: float = 0.0
    degraded_horizon_s: float = 0.0


@dataclass(frozen=True)
class CampaignStatistics:
    """Every scenario, plus the campaign-wide consistency check.

    Attributes
    ----------
    scenarios : tuple of ScenarioStatistics
        In campaign order.
    nees, nis : ConsistencyInterval
        Over the runs of the ``nominal`` scenario, the same population
        :attr:`ensemble` is measured on. Deliberately *not* pooled over the
        nominal-regime stretches of the fault scenarios: the regime tag records
        whether a fault is armed at that instant, not whether the estimate is
        still contaminated, so the recovery tail after a spoof or a bad fix
        carries the ``"nominal"`` label while the error is still kilometres and
        the innovation gate is still refusing honest fixes. NEES is quadratic,
        so a few thousand such samples swamp millions of healthy ones and the
        campaign gate reports a fault's size rather than the filter's honesty.
        The per-scenario intervals keep that behaviour visible, and it is the
        fault scenarios' own criteria that judge it.
    ensemble : tuple of EnsembleAxis
        The truth-derived spread against the reported 1σ, per RIC axis, over the
        nominal scenario. The only consistency evidence here that does not
        normalise the error by the covariance being judged. Empty when the
        campaign carried fewer than two runs with RIC records.
    runs, samples : int
        Totals.
    """

    scenarios: tuple[ScenarioStatistics, ...]
    nees: ConsistencyInterval
    nis: ConsistencyInterval
    ensemble: tuple[EnsembleAxis, ...]
    runs: int
    samples: int

    def of(self, scenario: str) -> ScenarioStatistics | None:
        """One scenario's statistics by name, or None when it was not flown."""
        for entry in self.scenarios:
            if entry.scenario == scenario:
                return entry
        return None


def _regime_summary(regime: str, runs: tuple[ScenarioRun, ...]) -> RegimeSummary:
    """Aggregate one regime across every run of a scenario."""
    position: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    ratio: list[np.ndarray] = []
    sma: list[np.ndarray] = []
    sma_ratio: list[np.ndarray] = []
    fixes = 0
    accepted = 0
    fine_fixes = 0
    fine_accepted = 0
    refusals: dict[str, int] = {}

    for run in runs:
        armed = run.mask(regime)
        if not armed.any():
            continue
        # An absent solution has no error. Counting the cycles where the filter
        # published nothing would report a dropped solution as zero error, which
        # inverts the meaning of the very case the outage scenarios exist for.
        live = armed & run.solution_valid
        position.append(run.pos_err_m[live])
        velocity.append(run.vel_err_mps[live])
        sigma = run.pos_sigma_m[live]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio.append(np.where(sigma > 0.0, run.pos_err_m[live] / sigma, np.nan))
            sma_err = np.abs(run.sma_err_m[live])
            sma_sig = run.sma_sigma_m[live]
            sma.append(sma_err)
            sma_ratio.append(np.where(sma_sig > 0.0, sma_err / sma_sig, np.nan))
        fixes += int(np.count_nonzero(run.fix_valid[armed]))
        accepted += int(np.count_nonzero(run.fix_accepted[armed]))
        fine = armed & np.array([q == "fine" for q in run.quality], dtype=bool)
        fine_fixes += int(np.count_nonzero(run.fix_valid[fine]))
        fine_accepted += int(np.count_nonzero(run.fix_accepted[fine]))
        for index in np.flatnonzero(armed):
            name = run.refusal[index]
            if name:
                refusals[name] = refusals.get(name, 0) + 1

    join = lambda parts: np.concatenate(parts) if parts else np.array([])  # noqa: E731
    return RegimeSummary(
        regime=regime,
        position=ErrorSummary.of(join(position)),
        velocity=ErrorSummary.of(join(velocity)),
        sigma_ratio=ErrorSummary.of(join(ratio)),
        fixes=fixes,
        accepted=accepted,
        refusals=refusals,
        fine_fixes=fine_fixes,
        fine_accepted=fine_accepted,
        sma=ErrorSummary.of(join(sma)),
        sma_sigma_ratio=ErrorSummary.of(join(sma_ratio)),
    )


def along_track_psd_from_one_orbit_error(
    sigma_along_track_m: float, period_s: float
) -> float:
    """NASA/TP-2018-219822 Eq. 2.88: the along-track SNC intensity from a one-orbit error.

    ``q_T = sigma_ds^2 / (3 T_p^3)`` [m^2/s^3] — the starting point the TP gives
    for tuning the along-track process noise (§2.2.4.2): take the period-folded
    ensemble along-track error one period out (``err_ric_m[:, 1]`` at
    ``t = T_p``), and this is the intensity whose secular term reproduces it.
    The flight side is ``OrbitEstimator.AccelPsdRtnM2PerS3[1]``. Returns 0 for
    a non-positive period.
    """
    if not (period_s > 0.0) or not np.isfinite(sigma_along_track_m):
        return 0.0
    return float(sigma_along_track_m) ** 2 / (3.0 * float(period_s) ** 3)


def _longest_gap_s(run: ScenarioRun) -> float:
    """Longest continuous stretch with no valid solution [s].

    Measured in cycles and multiplied by the period rather than differenced off
    the time column: a scenario's cadence is uniform by construction, and the
    cycle count is what the coast policy is actually written in terms of.
    """
    invalid = ~run.solution_valid
    if not invalid.any():
        return 0.0
    longest = current = 0
    for flag in invalid:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return float(longest) * run.cycle_period_s


def _longest_clean_lockout_s(run: ScenarioRun) -> float:
    """Longest unbroken stretch of delivered, un-faulted fixes the gate refused [s].

    The recovery number the per-scenario rejection *rate* cannot express. A rate
    averages a lockout over the whole arc, so a filter that refuses every honest
    fix for half an hour and one that scatters the same refusals across a day
    read alike; only the first is a filter that cannot get back. Counted over
    the nominal regime, so a spoof or a jam being correctly refused is not a
    lockout, and in cycles for the reason :func:`_longest_gap_s` gives.
    """
    armed = run.mask("nominal")
    refused = armed & run.fix_valid & ~run.fix_accepted
    if not refused.any():
        return 0.0
    longest = current = 0
    for flag in refused:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return float(longest) * run.cycle_period_s


def _run_mean_nominal(run: ScenarioRun, column: np.ndarray) -> float:
    """Mean of ``column`` over the run's nominal, initialised samples.

    NaN when the run had no such sample — a scenario armed end to end, or a run
    that never initialised. NaN is dropped by :func:`consistency_interval`
    rather than counted, because no information is not the same as zero error.
    """
    live = run.mask("nominal") & run.solution_valid
    values = column[live]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def summarise(campaign: Campaign) -> CampaignStatistics:
    """Reduce a loaded campaign to its statistics.

    Parameters
    ----------
    campaign : analysis.od.records.Campaign

    Returns
    -------
    CampaignStatistics
    """
    scenarios: list[ScenarioStatistics] = []

    for name in campaign.scenarios:
        runs = campaign.of(name)
        present = sorted({regime for run in runs for regime in run.regime})
        regimes = {regime: _regime_summary(regime, runs) for regime in present}

        nees_means = np.array([_run_mean_nominal(r, r.nees) for r in runs])
        nis_means = np.array([_run_mean_nominal(r, r.nis) for r in runs])

        worst = max(
            (
                summary.position.worst
                for summary in regimes.values()
                if np.isfinite(summary.position.worst)
            ),
            default=float("nan"),
        )
        first = runs[0]
        scenarios.append(
            ScenarioStatistics(
                scenario=name,
                intent=first.intent,
                runs=len(runs),
                samples=sum(r.samples for r in runs),
                cycle_period_s=first.cycle_period_s,
                fix_latency_s=first.fix_latency_s,
                duration_s=first.duration_s,
                regimes=regimes,
                nees=consistency_interval(nees_means, STATE_DIM),
                nis=consistency_interval(nis_means, MEASUREMENT_DIM),
                worst_pos_err_m=worst,
                solution_gap_s=max((_longest_gap_s(r) for r in runs), default=0.0),
                clean_lockout_s=max(
                    (_longest_clean_lockout_s(r) for r in runs), default=0.0
                ),
                degraded_horizon_s=first.degraded_horizon_s,
            )
        )

    # All three campaign-level checks are measured on `nominal` alone, and for
    # one reason: it is the only scenario whose samples are all drawn from the
    # distribution the covariance claims to describe. The fault scenarios share
    # its seed and so its trajectory, and a nominal-*regime* mask does not
    # recover a clean population from them — the tag falls the instant the
    # fault window closes, while the estimate is still contaminated and the
    # gate is still refusing good fixes. Their consistency is reported
    # per-scenario, where a reader can see it against the fault that caused it.
    nominal = campaign.of("nominal")
    nominal_nees = np.array([_run_mean_nominal(r, r.nees) for r in nominal])
    nominal_nis = np.array([_run_mean_nominal(r, r.nis) for r in nominal])

    return CampaignStatistics(
        scenarios=tuple(scenarios),
        nees=consistency_interval(nominal_nees, STATE_DIM),
        nis=consistency_interval(nominal_nis, MEASUREMENT_DIM),
        ensemble=ensemble_covariance(nominal),
        runs=len({r.run for r in campaign.runs}),
        samples=sum(r.samples for r in campaign.runs),
    )
