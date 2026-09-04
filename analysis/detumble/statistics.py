"""Sizing the campaign, and reading a handover time out of it.

What the campaign has to answer is not "how long does detumble take" but "what
time can a Safe-mode handover be scheduled at such that the vehicle is reliably
below :math:`\\omega_{exit}` by then". That is an upper tolerance bound on a high
quantile of the time-to-completion distribution, and it is what fixes the sample
size.

Why the sample size is what it is
---------------------------------
The distribution of the tail duration has no reason to be normal — it is set by
orbital geometry, is bounded below by the fast phase and has a long right tail
where the residual spin starts nearly parallel to the field. So the bound is
**distribution-free**: Wilks' order-statistic tolerance interval. For a one-sided
upper bound on the :math:`p` quantile at confidence :math:`\\gamma`, using the
:math:`k`-th largest of :math:`N` samples, the requirement is

.. math::
   \\sum_{i=0}^{k-1} \\binom{N}{i} (1-p)^i p^{N-i} \\;\\le\\; 1-\\gamma .

For :math:`p=\\gamma=0.95` this gives :math:`N=59` using the sample maximum
(:math:`k=1`) and :math:`N=93` using the second largest (:math:`k=2`). The
campaign is sized at the second order deliberately: a first-order bound *is* the
single worst run, so one pathological or mis-flown case sets the entire
requirement, with nothing to distinguish "the physics does this" from "the
harness hiccuped". At :math:`k=2` the bound survives one outlier and the outlier
is still visible in the report as the worst case.

Censoring
---------
A run whose flown arc ended before the completion predicate confirmed is
**right-censored**: its true time-to-exit is somewhere beyond the arc, not equal
to it. Order statistics computed over censored data are lower bounds on the true
ones, so :func:`summarise` records the censored count and the report fails on
it — a tolerance bound quoted from a campaign that did not finish is a number
pointing the wrong way.

References
----------
Wilks, "Determination of Sample Sizes for Setting Tolerance Limits," *Annals of
Mathematical Statistics* 12(1):91-96, 1941 [wilks1941] — the order-statistic
tolerance interval and its sample-size condition.
Conover, *Practical Nonparametric Statistics*, 3rd ed., Wiley, 1999, §3.3
[conover1999] — the distribution-free tolerance-limit tables this reproduces.
Avanzini & Giulietti, "Magnetic Detumbling of a Rigid Spacecraft," *JGCD*
35(4):1326-1334, 2012 [avanzini2012] — the asymptotic convergence the tail is.
Design doc §12 (analysis tools), §22.2 (margin reporting), §13 (Monte Carlo).

Units
-----
Seconds and degrees per second throughout; the orbit period used to express a
handover time in orbits is a parameter, not a constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analysis.detumble.records import RunRecord

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_HANDOVER_MARGIN",
    "DEFAULT_QUANTILE",
    "DetumbleStatistics",
    "spearman",
    "summarise",
    "wilks_bound",
    "wilks_sample_size",
]

#: Quantile and confidence the handover time is set from. 95/95 is the standard
#: aerospace one-sided tolerance pairing and matches how the rest of the
#: requirement baseline is argued (design doc §22.2).
DEFAULT_QUANTILE = 0.95
DEFAULT_CONFIDENCE = 0.95

#: Margin carried in the *proposed requirement value* over the measured bound.
#: The repo's standing rule is that a threshold is a requirement value and never
#: a transcribed measurement, and that a committed threshold within noise of its
#: own measurement is too tight; 20 % is the margin the neighbouring ADCS
#: requirements declare.
DEFAULT_HANDOVER_MARGIN = 0.20


def _coverage_tail(n: int, k: int, quantile: float) -> float:
    """:math:`\\sum_{i<k} \\binom{N}{i}(1-p)^i p^{N-i}` — the confidence shortfall."""
    return sum(
        math.comb(n, i) * (1.0 - quantile) ** i * quantile ** (n - i) for i in range(k)
    )


def wilks_sample_size(
    quantile: float = DEFAULT_QUANTILE,
    confidence: float = DEFAULT_CONFIDENCE,
    order: int = 2,
) -> int:
    """Smallest sample size supporting a one-sided distribution-free bound.

    Parameters
    ----------
    quantile : float, optional
        The quantile to bound, in (0, 1). ``0.95`` bounds the 95th percentile.
    confidence : float, optional
        Confidence in the bound, in (0, 1).
    order : int, optional
        Which order statistic the bound is read from: ``1`` is the sample
        maximum, ``2`` the second largest, and so on. Higher orders need more
        runs and are less sensitive to a single outlier.

    Returns
    -------
    int
        The smallest ``N`` for which the ``order``-th largest of ``N`` samples is
        a valid upper tolerance bound.

    Raises
    ------
    ValueError
        ``quantile`` or ``confidence`` outside (0, 1), or ``order`` below 1.

    Examples
    --------
    >>> wilks_sample_size(0.95, 0.95, order=1)
    59
    >>> wilks_sample_size(0.95, 0.95, order=2)
    93
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    n = order
    while _coverage_tail(n, order, quantile) > 1.0 - confidence:
        n += 1
    return n


def wilks_bound(
    samples,
    quantile: float = DEFAULT_QUANTILE,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, int]:
    """Tightest valid one-sided upper tolerance bound this sample supports.

    Searches for the largest ``order`` (i.e. the *tightest*, deepest-into-the-
    sample statistic) that still satisfies the coverage condition, and returns
    that order statistic.

    Parameters
    ----------
    samples : array_like
        The observed values. Must contain no censored entries — see the module
        docstring.
    quantile, confidence : float, optional
        As for :func:`wilks_sample_size`.

    Returns
    -------
    bound : float
        The bound, in the samples' units. ``nan`` when the sample is too small to
        support any bound at this quantile and confidence.
    order : int
        Which order statistic it came from (1 = the maximum). ``0`` when no bound
        is supported.
    """
    values = np.sort(np.asarray(samples, dtype=float))[::-1]
    n = values.size
    if n == 0 or _coverage_tail(n, 1, quantile) > 1.0 - confidence:
        return float("nan"), 0
    order = 1
    while order + 1 <= n and _coverage_tail(n, order + 1, quantile) <= 1.0 - confidence:
        order += 1
    return float(values[order - 1]), order


def spearman(x, y) -> float:
    """Spearman rank correlation, ``nan`` when either input has no spread.

    Rank rather than Pearson because the relationships of interest here are
    monotone but not linear — the tail duration grows sharply as the residual
    spin approaches the field line — and because ranks are insensitive to the
    heavy right tail that would dominate a Pearson coefficient.

    Parameters
    ----------
    x, y : array_like
        Equal-length samples.

    Returns
    -------
    float
        The coefficient in [-1, 1], or ``nan`` for a degenerate input.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass(frozen=True)
class DetumbleStatistics:
    """The campaign's distribution of time-to-completion, and what it supports.

    Every time is **seconds from B-dot engagement**, the same origin
    REQ-ACTL-001's fast-phase bound uses.

    Attributes
    ----------
    n_records, n_healthy, n_converged, n_censored : int
        Records read; of those, runs the harness completed; of those, runs that
        reached a confirmed completion; and runs that did not (right-censored).
    duration_s : float
        The arc each run flew, from engagement — the censoring horizon.
    median_s, p95_empirical_s, worst_s : float
        The empirical distribution of the converged runs.
    tolerance_bound_s : float
        The distribution-free upper bound on :attr:`quantile` at
        :attr:`confidence`.
    tolerance_order : int
        Which order statistic :attr:`tolerance_bound_s` was read from.
    quantile, confidence : float
        What the bound covers.
    required_n : int
        Sample size the order actually used needs — so the criterion is "this
        campaign supports the bound it quotes", not "this campaign reached some
        preferred size". When no bound is supported at all it is the first-order
        minimum, which is the target a short campaign has to reach.
    handover_s : float
        The proposed Safe-mode handover time: the tolerance bound plus
        :attr:`handover_margin`, rounded **up** to a whole orbit so the number is
        expressible in the units operations actually schedules in.
    handover_margin : float
        Fractional margin carried over the bound.
    orbit_period_s : float
        Period the handover time is rounded and expressed in.
    worst_fast_phase_deg_s : float
        Worst rate 200 s after engagement across all healthy runs — the
        REQ-ACTL-001 fast-phase bound, now measured over the dispersion instead
        of at one geometry.
    worst_peak_after_fast_phase_deg_s : float
        Worst rate at or after that instant, for the requirement's
        "shall not subsequently rise" clause.
    median_floor_deg_s : float
        Median across runs of the *lowest* rate the run ever reached [deg/s] —
        B-dot's achieved floor. Reported next to the completion threshold
        because the two being close is the condition under which completion
        stops being a time and becomes a coin flip on geometry.
    exit_threshold_deg_s : float
        The committed ``DetumbleExitRadps`` the runs were judged against
        [deg/s], read from the records rather than transcribed.
    n_re_excited : int
        Converged runs whose rate was back above the completion threshold at the
        end of the arc. The campaign holds the vehicle in DETUMBLE for the whole
        arc, which the flown CONOPS does not — the mode manager hands over on
        the predicate — so this is not a failure. It is the measurement that
        says *staying* in B-dot past completion is harmful, and therefore that
        the handover must be triggered by the predicate rather than scheduled
        loosely after it.
    correlations : dict
        Spearman rank correlation of the time-to-exit against each candidate
        driver, so the report can say *what* sets the tail rather than only how
        long it is.
    total_wall_s : float
        Summed per-run wall clock, for sizing the next campaign.
    n_below_entry : int
        Runs whose dispersed initial rate was below the deployment's
        ``DetumbleEnterRadps``. These never tumbled — the mode manager would not
        have engaged B-dot — so they measure nothing and must be zero for the
        campaign to be about detumble at all. Non-zero means the tip-off
        dispersion and the flight entry threshold have come apart, which is how
        11 of the first 18 runs of the Push 84 campaign came to report a
        completion time for a vehicle that arrived detumbled.
    """

    n_records: int
    n_healthy: int
    n_converged: int
    n_censored: int
    n_below_entry: int
    duration_s: float
    median_s: float
    p95_empirical_s: float
    worst_s: float
    tolerance_bound_s: float
    tolerance_order: int
    quantile: float
    confidence: float
    required_n: int
    handover_s: float
    handover_margin: float
    orbit_period_s: float
    worst_fast_phase_deg_s: float
    worst_peak_after_fast_phase_deg_s: float
    median_floor_deg_s: float
    exit_threshold_deg_s: float
    n_re_excited: int
    correlations: dict[str, float]
    total_wall_s: float

    @property
    def floor_margin(self) -> float:
        """Completion threshold as a multiple of the achieved floor.

        Below about 2 there is no headroom between what B-dot can deliver and
        what the predicate asks for, and whether a run confirms at all depends
        on geometry rather than on how long it is given.
        """
        if self.median_floor_deg_s <= 0.0:
            return float("inf")
        return self.exit_threshold_deg_s / self.median_floor_deg_s

    @property
    def handover_orbits(self) -> float:
        """:attr:`handover_s` expressed in orbits."""
        return self.handover_s / self.orbit_period_s


def summarise(
    records: list[RunRecord],
    *,
    orbit_period_s: float = 5677.0,
    quantile: float = DEFAULT_QUANTILE,
    confidence: float = DEFAULT_CONFIDENCE,
    handover_margin: float = DEFAULT_HANDOVER_MARGIN,
) -> DetumbleStatistics:
    """Reduce a campaign to its distribution and the handover time it supports.

    Parameters
    ----------
    records : list of RunRecord
        As loaded by :func:`analysis.detumble.records.load_records`.
    orbit_period_s : float, optional
        Orbit period of the flown scenario [s]; the reference vehicle's 500 km
        SSO is 5677 s. Used only to express and round the handover time.
    quantile, confidence : float, optional
        The tolerance bound's coverage.
    handover_margin : float, optional
        Fractional margin carried over the bound into the proposed handover time.

    Returns
    -------
    DetumbleStatistics

    Raises
    ------
    ValueError
        ``records`` is empty. An empty campaign has no distribution, and
        returning zeros would render as a very fast detumble.
    """
    if not records:
        raise ValueError("no campaign records to summarise")

    healthy = [r for r in records if r.healthy]
    converged = [r for r in healthy if r.converged]
    censored = [r for r in healthy if not r.converged]
    times = np.array([r.t_exit_s for r in converged], dtype=float)

    # The censoring horizon is the arc the runs that *failed* to converge were
    # given, because those are the only runs the horizon censors. It is a
    # minimum over the censored runs alone, not over all of them.
    #
    # Taking the minimum over every healthy run was correct only while the
    # driver flew a fixed arc. It now stops a settle window after completion, so
    # the shortest arc in a campaign is the *fastest* run's — and using that as
    # the horizon would report a censoring bound of a few hundred seconds for a
    # campaign whose censored runs each flew eight orbits, understating the arc
    # by two orders and making an uncensored campaign look severely censored.
    censored_arcs = [r.arc_s for r in censored if r.arc_s > 0.0]
    if censored_arcs:
        duration_s = float(min(censored_arcs))
    else:
        # Nothing was censored, so the horizon never bound. Report the longest
        # arc flown: it is the span over which the campaign can say anything.
        flown = [r.arc_s for r in healthy if r.arc_s > 0.0]
        duration_s = float(max(flown)) if flown else 0.0

    bound, order = (
        wilks_bound(times, quantile, confidence) if times.size else (float("nan"), 0)
    )
    handover = (
        math.ceil(bound * (1.0 + handover_margin) / orbit_period_s) * orbit_period_s
        if math.isfinite(bound)
        else float("nan")
    )

    predictors = {
        "initial rate magnitude [deg/s]": [r.rate_initial_deg_s for r in converged],
        "rate at end of fast phase [deg/s]": [
            r.rate_at_fast_phase_deg_s for r in converged
        ],
        # **|sin| of the spin/field angle, not the angle.** B-dot's torque is
        # m x B with m proportional to the body-frame dB/dt, so what the law can
        # remove is the component of omega *perpendicular* to B — and the
        # perpendicular fraction is |sin(theta)|, which is symmetric about 90
        # degrees. A rank correlation against the raw angle is therefore
        # structurally blind to the effect: 0 and 180 degrees are both fully
        # aligned and both terrible, so the relationship is not monotone in
        # theta and Spearman reports approximately nothing.
        #
        # Measured on the 93-run campaign, against the rate 200 s after
        # engagement: raw angle -0.07, |sin| of the same angle **-0.65**. The
        # report existed to say what sets the tail and was reporting that
        # geometry did not, because it asked in a coordinate the physics is not
        # monotone in.
        "perpendicular spin fraction at engagement |sin|": [
            abs(math.sin(math.radians(r.initial_spin_field_angle_deg)))
            for r in converged
        ],
        "perpendicular spin fraction after fast phase |sin|": [
            abs(math.sin(math.radians(r.fast_phase_spin_field_angle_deg)))
            for r in converged
        ],
        "RAAN offset [deg]": [
            float(r.dispersion.get("delta_raan_deg", math.nan)) for r in converged
        ],
        "argument-of-latitude offset [deg]": [
            float(r.dispersion.get("delta_arglat_deg", math.nan)) for r in converged
        ],
        "epoch offset [s]": [
            float(r.dispersion.get("delta_epoch_s", math.nan)) for r in converged
        ],
    }
    correlations = {
        name: spearman(values, times) for name, values in predictors.items()
    }

    return DetumbleStatistics(
        n_records=len(records),
        n_below_entry=sum(
            1
            for r in records
            # NaN compares false, so a shard predating the recorded threshold is
            # not counted as a violation it cannot be judged for.
            if r.rate_initial_deg_s < r.enter_threshold_deg_s
        ),
        n_healthy=len(healthy),
        n_converged=len(converged),
        n_censored=len(censored),
        duration_s=duration_s,
        median_s=float(np.median(times)) if times.size else float("nan"),
        p95_empirical_s=float(np.percentile(times, 95.0))
        if times.size
        else float("nan"),
        worst_s=float(np.max(times)) if times.size else float("nan"),
        tolerance_bound_s=bound,
        tolerance_order=order,
        quantile=quantile,
        confidence=confidence,
        required_n=wilks_sample_size(quantile, confidence, order=max(order, 1)),
        handover_s=handover,
        handover_margin=handover_margin,
        orbit_period_s=orbit_period_s,
        worst_fast_phase_deg_s=(
            max(r.rate_at_fast_phase_deg_s for r in healthy)
            if healthy
            else float("nan")
        ),
        worst_peak_after_fast_phase_deg_s=(
            max(r.peak_rate_after_fast_phase_deg_s for r in healthy)
            if healthy
            else float("nan")
        ),
        median_floor_deg_s=(
            float(np.median([r.rate_min_deg_s for r in healthy]))
            if healthy
            else float("nan")
        ),
        exit_threshold_deg_s=(
            float(np.median([r.exit_threshold_deg_s for r in records]))
        ),
        n_re_excited=sum(
            1 for r in converged if r.rate_final_deg_s > r.exit_threshold_deg_s
        ),
        correlations=correlations,
        total_wall_s=float(sum(r.wall_s for r in records)),
    )
