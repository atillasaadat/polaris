"""The detumble campaign's statistics and report layer (design doc §23.2).

Deliberately built on **synthetic records**, not on a campaign run: the whole
point of the C++/Python split is that the statistics can be exercised without
flying orbits of closed-loop simulation, and a test that needs a four-hour
campaign to run is a test nobody runs. What is under test here is the sample-size
arithmetic, the order-statistic bound, the censoring policy and the criteria the
report is built from — everything the campaign's *conclusions* rest on.

The Wilks sample sizes are checked against the published tables
(Conover, *Practical Nonparametric Statistics*, §3.3 [conover1999]), which is
what makes these closed-form cases rather than a re-derivation of the code by
the test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from analysis.detumble.records import load_records
from analysis.detumble.report import FAST_PHASE_BOUND_DEG_S, detumble_report
from analysis.detumble.statistics import (
    spearman,
    summarise,
    wilks_bound,
    wilks_sample_size,
)

ORBIT_S = 5677.0


def make_record(
    index: int,
    *,
    t_exit_s: float,
    healthy: bool = True,
    fast_phase_deg_s: float = 3.0,
    peak_after_deg_s: float = 3.0,
    rate_initial_deg_s: float = 5.0,
    spin_field_deg: float = 30.0,
    arc_s: float = 8.0 * ORBIT_S,
) -> dict:
    """One synthetic JSONL record, in the driver's own schema."""
    profile_t = np.arange(0.0, arc_s, 600.0)
    return {
        "exit_threshold_deg_s": 0.4984,
        "confirm_cycles": 50,
        "run_index": index,
        "seed": 1000 + index,
        "config_hash": "deadbeef" * 8,
        "scenario": "leo-sso-500km",
        "dispersion": {
            "rate_deg_s": rate_initial_deg_s,
            "rate_axis_body": [0.0, 0.0, 1.0],
            "attitude_wxyz": [1.0, 0.0, 0.0, 0.0],
            "delta_raan_deg": (index * 37) % 360,
            "delta_arglat_deg": (index * 53) % 360,
            "delta_epoch_s": float(index * 811 % 86164),
        },
        "healthy": healthy,
        "note": "" if healthy else "fork failed",
        "t_engage_s": 40.0,
        "t_exit_s": t_exit_s,
        "t_first_below_s": max(t_exit_s - 5.0, -1.0),
        "rate_initial_deg_s": rate_initial_deg_s,
        "rate_at_fast_phase_deg_s": fast_phase_deg_s,
        "rate_final_deg_s": 0.3,
        "rate_min_deg_s": 0.3,
        "peak_rate_after_fast_phase_deg_s": peak_after_deg_s,
        "initial_spin_field_angle_deg": 45.0,
        "fast_phase_spin_field_angle_deg": spin_field_deg,
        "wall_s": 1500.0,
        "profile_t_s": profile_t.tolist(),
        "profile_rate_deg_s": np.full(profile_t.size, 1.0).tolist(),
    }


def write_campaign(
    tmp_path: Path, records: list[dict], name: str = "runs.jsonl"
) -> Path:
    """Write records as JSONL, shuffled — the driver's threads interleave."""
    path = tmp_path / name
    shuffled = records[::2] + records[1::2]
    path.write_text("".join(json.dumps(r) + "\n" for r in shuffled))
    return path


# ----------------------------------------------------------------------
# Sample sizing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantile", "confidence", "order", "expected"),
    [
        (0.95, 0.95, 1, 59),  # the classic Wilks first-order 95/95 sample size
        (0.95, 0.95, 2, 93),
        (0.95, 0.90, 1, 45),
        (0.90, 0.90, 1, 22),
        (0.99, 0.95, 1, 299),
    ],
)
def test_wilks_sample_size_matches_the_published_tables(
    quantile: float, confidence: float, order: int, expected: int
) -> None:
    assert wilks_sample_size(quantile, confidence, order=order) == expected


def test_wilks_sample_size_is_the_smallest_n_that_works() -> None:
    """N-1 must fail the coverage condition, or the answer is not minimal."""
    n = wilks_sample_size(0.95, 0.95, order=1)
    assert 0.95**n <= 0.05
    assert 0.95 ** (n - 1) > 0.05


@pytest.mark.parametrize(
    ("quantile", "confidence", "order"),
    [(0.0, 0.95, 1), (1.0, 0.95, 1), (0.95, 0.0, 1), (0.95, 1.0, 1), (0.95, 0.95, 0)],
)
def test_wilks_sample_size_refuses_impossible_arguments(
    quantile: float, confidence: float, order: int
) -> None:
    with pytest.raises(ValueError):
        wilks_sample_size(quantile, confidence, order=order)


# ----------------------------------------------------------------------
# The bound
# ----------------------------------------------------------------------


def test_wilks_bound_at_exactly_59_samples_is_the_maximum() -> None:
    """59 is the first-order 95/95 size, so the bound is the sample maximum."""
    samples = np.arange(1.0, 60.0)
    bound, order = wilks_bound(samples, 0.95, 0.95)
    assert order == 1
    assert bound == pytest.approx(59.0)


def test_wilks_bound_tightens_as_the_sample_grows() -> None:
    """More runs buy a deeper order statistic, i.e. a bound below the worst run."""
    samples = np.arange(1.0, 94.0)
    bound, order = wilks_bound(samples, 0.95, 0.95)
    assert order == 2
    assert bound == pytest.approx(92.0)
    assert bound < samples.max()


def test_wilks_bound_refuses_an_undersized_sample() -> None:
    """58 samples cannot support a 95/95 bound at any order — say so, don't guess."""
    bound, order = wilks_bound(np.arange(1.0, 59.0), 0.95, 0.95)
    assert order == 0
    assert math.isnan(bound)


def test_wilks_bound_is_order_insensitive_to_input_ordering() -> None:
    rng = np.random.default_rng(20260807)
    samples = rng.uniform(0.0, 1.0, size=93)
    assert wilks_bound(samples) == wilks_bound(rng.permutation(samples))


# ----------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------


def test_spearman_is_one_on_a_monotone_nonlinear_relation() -> None:
    """Rank correlation, not Pearson: the tail is monotone but strongly curved."""
    x = np.linspace(0.1, 2.0, 40)
    assert spearman(x, np.exp(5.0 * x)) == pytest.approx(1.0)


def test_spearman_is_nan_without_spread() -> None:
    assert math.isnan(spearman(np.ones(10), np.arange(10.0)))


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


def test_load_records_sorts_by_run_index(tmp_path: Path) -> None:
    """Worker threads interleave writes; pairing a record with its draw needs order."""
    path = write_campaign(
        tmp_path, [make_record(i, t_exit_s=100.0 * i) for i in range(10)]
    )
    assert [r.run_index for r in load_records(path)] == list(range(10))


def test_load_records_reads_a_directory_of_shards(tmp_path: Path) -> None:
    write_campaign(
        tmp_path, [make_record(i, t_exit_s=1.0) for i in range(3)], "a.jsonl"
    )
    write_campaign(
        tmp_path, [make_record(i, t_exit_s=1.0) for i in range(3, 6)], "b.jsonl"
    )
    assert len(load_records(tmp_path)) == 6


def test_load_records_refuses_a_malformed_line(tmp_path: Path) -> None:
    """A silently dropped record shrinks the sample the confidence rests on."""
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(make_record(0, t_exit_s=1.0)) + "\n{not json}\n")
    with pytest.raises(ValueError, match="malformed campaign record"):
        load_records(path)


def test_load_records_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path / "nothing.jsonl")


# ----------------------------------------------------------------------
# The summary
# ----------------------------------------------------------------------


@pytest.fixture
def healthy_campaign(tmp_path: Path) -> Path:
    """93 converged runs with a known, deliberately skewed time distribution."""
    rng = np.random.default_rng(20260807)
    times = np.sort(rng.lognormal(mean=math.log(2.0 * ORBIT_S), sigma=0.5, size=93))
    return write_campaign(
        tmp_path,
        [
            make_record(
                i,
                t_exit_s=float(t),
                spin_field_deg=float(90.0 - 60.0 * (t / times.max())),
                rate_initial_deg_s=float(rng.uniform(2.0, 5.0)),
            )
            for i, t in enumerate(times)
        ],
    )


def test_summarise_reports_the_distribution_and_the_bound(
    healthy_campaign: Path,
) -> None:
    stats = summarise(load_records(healthy_campaign), orbit_period_s=ORBIT_S)
    assert stats.n_records == 93
    assert stats.n_converged == 93
    assert stats.n_censored == 0
    assert stats.tolerance_order == 2
    # The 95/95 bound sits between the empirical 95th percentile and the worst
    # run: it is the second largest of 93, by construction.
    assert stats.p95_empirical_s <= stats.tolerance_bound_s <= stats.worst_s
    assert stats.median_s < stats.p95_empirical_s


def test_summarise_proposes_a_handover_above_the_bound_and_on_an_orbit(
    healthy_campaign: Path,
) -> None:
    """The proposal carries margin and is expressible in whole orbits."""
    stats = summarise(load_records(healthy_campaign), orbit_period_s=ORBIT_S)
    assert stats.handover_s >= stats.tolerance_bound_s * (1.0 + stats.handover_margin)
    assert stats.handover_s % ORBIT_S == pytest.approx(0.0)
    assert stats.handover_orbits == pytest.approx(round(stats.handover_orbits))


def test_summarise_recovers_the_planted_driver(healthy_campaign: Path) -> None:
    """The synthetic campaign plants a monotone spin/field relation; find it."""
    stats = summarise(load_records(healthy_campaign), orbit_period_s=ORBIT_S)
    assert stats.correlations["spin/field angle after fast phase [deg]"] < -0.9
    assert abs(stats.correlations["initial rate magnitude [deg/s]"]) < 0.5


def test_summarise_separates_censoring_from_harness_failure(tmp_path: Path) -> None:
    """A run that ran out of arc and a run that never flew are different facts."""
    records = [make_record(i, t_exit_s=1000.0) for i in range(10)]
    records.append(make_record(10, t_exit_s=-1.0))  # censored: arc ended first
    records.append(make_record(11, t_exit_s=-1.0, healthy=False))  # harness failure
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    assert stats.n_records == 12
    assert stats.n_healthy == 11
    assert stats.n_converged == 10
    assert stats.n_censored == 1


def test_summarise_refuses_an_empty_campaign() -> None:
    with pytest.raises(ValueError, match="no campaign records"):
        summarise([])


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def test_report_passes_a_clean_campaign(healthy_campaign: Path) -> None:
    records = load_records(healthy_campaign)
    report = detumble_report(
        summarise(records, orbit_period_s=ORBIT_S), str(healthy_campaign)
    )
    assert report.passes, report.format_text()
    assert report.by_requirement("REQ-ACTL-001")


def test_report_fails_on_a_missed_fast_phase_bound(tmp_path: Path) -> None:
    """One run over REQ-ACTL-001's bound fails the campaign, not just that run."""
    records = [make_record(i, t_exit_s=1000.0) for i in range(93)]
    records[7]["rate_at_fast_phase_deg_s"] = FAST_PHASE_BOUND_DEG_S + 0.1
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    report = detumble_report(stats, "synthetic")
    assert not report.passes
    assert any(c.requirement == "REQ-ACTL-001" for c in report.failures())


def test_report_fails_on_re_excitation_after_the_fast_phase(tmp_path: Path) -> None:
    records = [make_record(i, t_exit_s=1000.0) for i in range(93)]
    records[3]["peak_rate_after_fast_phase_deg_s"] = FAST_PHASE_BOUND_DEG_S + 1.0
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    assert not detumble_report(stats, "synthetic").passes


def test_report_fails_and_warns_on_a_censored_campaign(tmp_path: Path) -> None:
    """A bound quoted from an unfinished campaign points the wrong way."""
    records = [make_record(i, t_exit_s=1000.0) for i in range(93)]
    records[0]["t_exit_s"] = -1.0
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    report = detumble_report(stats, "synthetic")
    assert not report.passes
    assert any("censored" in w for w in report.warnings)


def test_report_fails_an_undersized_campaign(tmp_path: Path) -> None:
    """Ten runs cannot support a 95/95 bound; the report must say so, not round up."""
    records = [make_record(i, t_exit_s=1000.0) for i in range(10)]
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    report = detumble_report(stats, "synthetic")
    assert not report.passes
    assert math.isnan(stats.tolerance_bound_s)


def test_report_always_warns_that_the_handover_is_a_proposal(
    healthy_campaign: Path,
) -> None:
    """The number must never read as a requirement it has not been made into."""
    stats = summarise(load_records(healthy_campaign), orbit_period_s=ORBIT_S)
    report = detumble_report(stats, "synthetic")
    assert any("proposal, not a requirement" in w for w in report.warnings)
    assert not any(c.name.startswith("Safe-mode handover") for c in report.criteria)


def test_report_renders_self_contained_text(healthy_campaign: Path) -> None:
    stats = summarise(load_records(healthy_campaign), orbit_period_s=ORBIT_S)
    text = detumble_report(stats, "synthetic").format_text()
    assert "REQ-ACTL-001" in text
    assert "PROPOSED" in text
    assert "Assumptions in force" in text


# ----------------------------------------------------------------------
# Plots and CLI
# ----------------------------------------------------------------------


def test_write_all_produces_every_figure_and_the_report(
    healthy_campaign: Path, tmp_path: Path
) -> None:
    from analysis.detumble.plots import write_all

    records = load_records(healthy_campaign)
    stats = summarise(records, orbit_period_s=ORBIT_S)
    report = detumble_report(stats, "synthetic")
    written = write_all(records, stats, report.format_text(), tmp_path / "figs")
    assert [p.name for p in written] == [
        "detumble_ensemble.png",
        "detumble_time_to_exit.png",
        "detumble_drivers.png",
        "detumble_mc_report.txt",
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_cli_exits_zero_on_a_clean_campaign(
    healthy_campaign: Path, tmp_path: Path
) -> None:
    from analysis.detumble.__main__ import main

    assert main([str(healthy_campaign), "--out", str(tmp_path / "out")]) == 0


def test_cli_exits_nonzero_on_an_undersized_campaign(tmp_path: Path) -> None:
    from analysis.detumble.__main__ import main

    path = write_campaign(
        tmp_path, [make_record(i, t_exit_s=1000.0) for i in range(10)]
    )
    assert main([str(path), "--no-plots"]) == 1


# ----------------------------------------------------------------------
# The floor and the re-excitation warning (both found by the pilot campaign)
# ----------------------------------------------------------------------


def test_floor_margin_warns_when_the_threshold_sits_on_the_floor(
    tmp_path: Path,
) -> None:
    """A predicate B-dot only just reaches is a bound on luck, not on time."""
    records = [make_record(i, t_exit_s=1000.0) for i in range(93)]
    for r in records:
        r["rate_min_deg_s"] = 0.40  # threshold 0.4984 -> 1.25x headroom
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    assert stats.floor_margin < 2.0
    assert any("decided by geometry" in w for w in detumble_report(stats, "x").warnings)


def test_ample_floor_headroom_raises_no_warning(tmp_path: Path) -> None:
    records = [make_record(i, t_exit_s=1000.0) for i in range(93)]
    for r in records:
        r["rate_min_deg_s"] = 0.02
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    assert stats.floor_margin > 2.0
    assert not any(
        "decided by geometry" in w for w in detumble_report(stats, "x").warnings
    )


def test_re_excitation_is_counted_and_warned_but_does_not_fail(tmp_path: Path) -> None:
    """Staying in DETUMBLE past completion is the campaign's artifact, not a FAIL."""
    records = [make_record(i, t_exit_s=1000.0) for i in range(93)]
    for r in records[:4]:
        r["rate_final_deg_s"] = 1.3  # back above the 0.4984 threshold
    stats = summarise(
        load_records(write_campaign(tmp_path, records)), orbit_period_s=ORBIT_S
    )
    report = detumble_report(stats, "x")
    assert stats.n_re_excited == 4
    assert any("triggered by" in w for w in report.warnings)
    assert report.passes
