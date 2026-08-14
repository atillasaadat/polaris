"""The orbit-OD campaign statistics (design doc §8.3, §23.2).

Synthetic campaigns with the answer known in advance, for the same reason the
detumble suite uses them: what is under test is the sample-size arithmetic, the
independence accounting, the censoring policy and the covariance checks — none
of which need a week of propagated orbits to exercise, and all of which the
campaign's conclusions rest on.

The three covariance checks the package carries are split across two lanes.
NEES and NIS are self-normalised and live in :mod:`analysis.od.statistics`; the
truth-derived ensemble spread lives in :mod:`analysis.od.ensemble`. The tests
below cover both, and in particular the case that separates them: a filter
whose error and whose covariance are wrong by the same factor, which NEES
cannot see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import od_sample, write_od_shard

from analysis.od.ensemble import ensemble_covariance
from analysis.od.records import load_campaign
from analysis.od.statistics import (
    MEASUREMENT_DIM,
    STATE_DIM,
    ErrorSummary,
    consistency_interval,
    summarise,
)


def campaign_of(tmp_path: Path, groups, name: str = "shard.jsonl"):
    """Write one shard and read it back."""
    return load_campaign(write_od_shard(tmp_path / name, groups))


# --------------------------------------------------------------------------
# Consistency intervals
# --------------------------------------------------------------------------


def test_the_interval_brackets_the_expected_nees() -> None:
    """A consistent filter averages the state dimension and sits inside."""
    interval = consistency_interval(np.full(20, float(STATE_DIM)), STATE_DIM)

    assert interval.consistent
    assert not interval.optimistic
    assert interval.lower < STATE_DIM < interval.upper
    assert interval.samples == 20


def test_the_interval_narrows_as_runs_accumulate() -> None:
    """Width falls as 1/sqrt(N) — the reason the campaign demands ten runs.

    Asserted as a ratio rather than against tabulated widths so the test states
    the property the sample-count policy is built on, not a transcription of
    the chi-square table.
    """
    narrow = consistency_interval(np.full(100, 6.0), STATE_DIM)
    wide = consistency_interval(np.full(4, 6.0), STATE_DIM)

    assert (narrow.upper - narrow.lower) < 0.25 * (wide.upper - wide.lower)


def test_a_factor_of_two_optimistic_filter_is_caught_at_ten_runs() -> None:
    """The failure the ten-run floor exists to make visible.

    A covariance half the size it should be doubles the NEES. At ten runs that
    is outside the interval and flagged in the *optimistic* direction, which is
    the unsafe one.
    """
    interval = consistency_interval(np.full(10, 2.0 * STATE_DIM), STATE_DIM)

    assert not interval.consistent
    assert interval.optimistic


def test_a_pessimistic_filter_is_outside_but_not_optimistic() -> None:
    """Below the interval is safe and must not be reported as the unsafe case."""
    interval = consistency_interval(np.full(10, 0.5 * STATE_DIM), STATE_DIM)

    assert not interval.consistent
    assert not interval.optimistic


def test_runs_that_judged_nothing_are_dropped_not_counted() -> None:
    """No information is not zero error.

    A run that never initialised contributes NaN, and counting it as a sample
    would narrow the interval on evidence that does not exist.
    """
    values = np.array([6.0, 6.0, np.nan, 6.0])

    assert consistency_interval(values, STATE_DIM).samples == 3


def test_an_empty_set_yields_an_untestable_interval() -> None:
    """Nothing to test reports NaN rather than a vacuous pass."""
    interval = consistency_interval(np.array([]), MEASUREMENT_DIM)

    assert interval.samples == 0
    assert np.isnan(interval.mean)
    assert not interval.consistent


# --------------------------------------------------------------------------
# Independence: scenarios within a run are replicates, not samples
# --------------------------------------------------------------------------


def test_nine_scenarios_of_one_run_count_as_one_sample(tmp_path: Path) -> None:
    """The seed is a function of the run alone, so its scenarios share a trajectory.

    Pooling them flat would inflate N ninefold, narrow the chi-square interval
    by about a factor of three, and let a two-run campaign satisfy a gate
    written to demand ten. The regression this pins is exactly that.
    """
    scenarios = [
        "nominal",
        "outage",
        "spoof",
        "jam",
        "clock_jump",
        "radius_jump",
        "sigma_degrade",
        "spoof_ramp",
        "latency",
    ]
    stats = summarise(
        campaign_of(
            tmp_path,
            [(0, name, [od_sample(i) for i in range(20)]) for name in scenarios],
        )
    )

    assert len(stats.scenarios) == 9
    assert stats.runs == 1
    assert stats.nees.samples == 1
    assert stats.nis.samples == 1


def test_independent_runs_each_count_once(tmp_path: Path) -> None:
    """Runs are the independent unit; each contributes exactly one sample."""
    for run in range(6):
        write_od_shard(
            tmp_path / f"shard_{run}.jsonl",
            [
                (run, name, [od_sample(i) for i in range(10)])
                for name in ("nominal", "outage")
            ],
        )

    stats = summarise(load_campaign(tmp_path))

    assert stats.runs == 6
    assert stats.nees.samples == 6


# --------------------------------------------------------------------------
# Regime summaries and the censoring policy
# --------------------------------------------------------------------------


def test_cycles_with_no_solution_are_excluded_from_the_error_stats(
    tmp_path: Path,
) -> None:
    """An absent estimate has no error.

    Counting it as zero would report a dropped solution as perfect accuracy,
    which inverts the meaning of the outage scenarios that exist to produce it.
    """
    rows = [od_sample(i, regime="outage", pos_err_m=4.0) for i in range(5)]
    rows += [
        od_sample(i, regime="outage", solution_valid=0, pos_err_m=0.0)
        for i in range(5, 15)
    ]
    stats = summarise(campaign_of(tmp_path, [(0, "outage", rows)]))

    summary = stats.of("outage").regimes["outage"]
    assert summary.position.samples == 5
    assert summary.position.worst == 4.0


def test_the_rejection_rate_counts_delivered_against_folded_in(tmp_path: Path) -> None:
    """Fixes refused over fixes delivered, per regime."""
    rows = [od_sample(i) for i in range(8)]
    rows += [
        od_sample(i, fix_accepted=0, refusal="innovation_gate") for i in range(8, 10)
    ]
    stats = summarise(campaign_of(tmp_path, [(0, "nominal", rows)]))

    summary = stats.of("nominal").regimes["nominal"]
    assert (summary.fixes, summary.accepted) == (10, 8)
    assert summary.rejection_rate == pytest.approx(0.2)
    assert summary.refusals == {"innovation_gate": 2}


def test_which_refusal_fired_is_kept_apart(tmp_path: Path) -> None:
    """A fault caught at the trust boundary and one caught a layer in differ.

    They are indistinguishable in any count that records only *that* a refusal
    happened, and the report's band criterion is built on the distinction.
    """
    rows = [
        od_sample(i, regime="radius_jump", fix_accepted=0, refusal="fix_implausible")
        for i in range(6)
    ]
    rows += [
        od_sample(i, regime="radius_jump", fix_accepted=0, refusal="innovation_gate")
        for i in range(6, 8)
    ]
    stats = summarise(campaign_of(tmp_path, [(0, "radius_jump", rows)]))

    assert stats.of("radius_jump").regimes["radius_jump"].refusals == {
        "fix_implausible": 6,
        "innovation_gate": 2,
    }


def test_the_solution_gap_is_the_longest_continuous_outage(tmp_path: Path) -> None:
    """Longest run of invalid cycles, in seconds — not the total of them."""
    rows = [od_sample(i) for i in range(3)]
    rows += [od_sample(i, solution_valid=0) for i in range(3, 5)]  # 2 cycles
    rows += [od_sample(i) for i in range(5, 7)]
    rows += [od_sample(i, solution_valid=0) for i in range(7, 12)]  # 5 cycles
    rows += [od_sample(i) for i in range(12, 14)]
    stats = summarise(campaign_of(tmp_path, [(0, "outage", rows)]))

    assert stats.of("outage").solution_gap_s == 50.0


def test_a_filter_that_never_drops_reports_no_gap(tmp_path: Path) -> None:
    """Zero means published throughout, and must not be confused with unknown."""
    stats = summarise(
        campaign_of(tmp_path, [(0, "nominal", [od_sample(i) for i in range(9)])])
    )

    assert stats.of("nominal").solution_gap_s == 0.0


def test_the_error_summary_is_quantiles_not_moments() -> None:
    """The distribution is bounded below and long-tailed; a mean describes neither."""
    summary = ErrorSummary.of(np.array([1.0] * 99 + [100.0]))

    assert summary.median == 1.0
    assert summary.worst == 100.0
    assert summary.samples == 100


def test_an_empty_selection_summarises_to_nan_not_zero() -> None:
    """A regime that was never armed measured nothing; zero would be a claim."""
    summary = ErrorSummary.of(np.array([]))

    assert summary.samples == 0
    assert np.isnan(summary.median)


# --------------------------------------------------------------------------
# The ensemble covariance: the check that consults truth
# --------------------------------------------------------------------------


def ric_runs(tmp_path: Path, errors_by_run, sigma: float = 1.0):
    """A nominal campaign whose RIC error is dictated per run."""
    for run, error in enumerate(errors_by_run):
        write_od_shard(
            tmp_path / f"shard_{run}.jsonl",
            [
                (
                    run,
                    "nominal",
                    [
                        od_sample(
                            i,
                            err_ric_m=[error, error, error],
                            sigma_ric_m=[sigma, sigma, sigma],
                        )
                        for i in range(10)
                    ],
                )
            ],
        )
    return load_campaign(tmp_path).of("nominal")


def test_a_matching_covariance_reports_a_ratio_of_one(tmp_path: Path) -> None:
    """Errors of ±1 about zero against a claimed 1σ of 1: the honest filter."""
    axes = ensemble_covariance(ric_runs(tmp_path, [1.0, -1.0, 1.0, -1.0]))

    assert [a.axis for a in axes] == ["radial", "in_track", "cross_track"]
    for axis in axes:
        assert axis.ratio == 1.0
        assert axis.runs == 4


def test_a_covariance_half_the_true_spread_is_caught(tmp_path: Path) -> None:
    """The optimistic filter: the spread is twice what it admits to."""
    axes = ensemble_covariance(ric_runs(tmp_path, [2.0, -2.0, 2.0, -2.0], sigma=1.0))

    assert all(a.ratio == 2.0 for a in axes)


def test_a_bias_is_caught_rather_than_subtracted_out(tmp_path: Path) -> None:
    """The spread is taken about zero, not about the sample mean.

    Every run wrong by the same +2 m is a filter with a real bias. Centring on
    the sample mean would remove it and report a ratio of zero — a perfectly
    consistent filter that is two metres off every single time.
    """
    axes = ensemble_covariance(ric_runs(tmp_path, [2.0, 2.0, 2.0, 2.0], sigma=1.0))

    assert all(a.ratio == 2.0 for a in axes)


def test_one_run_is_not_an_ensemble(tmp_path: Path) -> None:
    """A spread cannot be estimated from a single sample."""
    assert ensemble_covariance(ric_runs(tmp_path, [1.0])) == ()


def test_stale_shards_without_ric_columns_are_excluded(tmp_path: Path) -> None:
    """A campaign predating the decomposition yields no ensemble, not a zero one."""
    for run in range(4):
        rows = [od_sample(i) for i in range(10)]
        for row in rows:
            del row["err_ric_m"]
            del row["sigma_ric_m"]
        write_od_shard(tmp_path / f"shard_{run}.jsonl", [(run, "nominal", rows)])

    assert ensemble_covariance(load_campaign(tmp_path).of("nominal")) == ()


def test_the_ensemble_is_measured_on_the_nominal_stretch_only(tmp_path: Path) -> None:
    """Fault stretches are not drawn from the distribution the covariance describes.

    A spoof walking the estimate kilometres off is not evidence the covariance
    is wrong; including it would widen the measured spread with samples the
    filter never claimed to cover.
    """
    for run in range(4):
        rows = [
            od_sample(i, err_ric_m=[1.0, 1.0, 1.0], sigma_ric_m=[1.0, 1.0, 1.0])
            for i in range(10)
        ]
        rows += [
            od_sample(
                i,
                regime="spoof",
                err_ric_m=[500.0, 500.0, 500.0],
                sigma_ric_m=[1.0, 1.0, 1.0],
            )
            for i in range(10, 20)
        ]
        write_od_shard(tmp_path / f"shard_{run}.jsonl", [(run, "nominal", rows)])

    axes = ensemble_covariance(load_campaign(tmp_path).of("nominal"))

    assert all(a.ratio == 1.0 for a in axes)


def test_the_campaign_statistics_carry_the_ensemble(tmp_path: Path) -> None:
    """`summarise` wires the nominal scenario's ensemble into the verdict object."""
    for run in range(3):
        write_od_shard(
            tmp_path / f"shard_{run}.jsonl",
            [
                (
                    run,
                    "nominal",
                    [
                        od_sample(
                            i, err_ric_m=[1.0, 1.0, 1.0], sigma_ric_m=[1.0, 1.0, 1.0]
                        )
                        for i in range(10)
                    ],
                )
            ],
        )

    stats = summarise(load_campaign(tmp_path))

    assert len(stats.ensemble) == 3
    assert all(a.ratio == 1.0 for a in stats.ensemble)
