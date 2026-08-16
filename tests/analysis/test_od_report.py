"""The orbit-OD campaign verdict (design doc §8.3, §9.2, §23.2).

The line this suite defends is the one the report's own docstring draws: what
may be a criterion here and what may only be a measurement. No requirement
writes a number on the filter's position accuracy, so an accuracy threshold
invented here and then passed against would be the threshold-tuned-to-its-own-
measurement defect the review-lessons catalogue names. These tests assert that
the criteria are the claims with a threshold this campaign did not choose — the
chi-square intervals, the gate's own configured rate, the fault policy — and
that the unbounded results stay warnings.

Assertions are on the structured :class:`~analysis.common.report.AnalysisReport`
throughout. Never on rendered text: the report object is the verdict and the
text is a rendering of it.
"""

from __future__ import annotations

from pathlib import Path

from conftest import od_sample, write_od_shard

from analysis.od.records import load_campaign
from analysis.od.report import (
    ENSEMBLE_RATIO_MAX,
    MIN_CAMPAIGN_RUNS,
    NOMINAL_REJECTION_CEILING,
    od_report,
)
from analysis.od.statistics import summarise


def report_of(tmp_path: Path, groups_by_run) -> object:
    """Fly a synthetic campaign through the whole loader/statistics/report path."""
    for run, groups in enumerate(groups_by_run):
        write_od_shard(tmp_path / f"shard_{run}.jsonl", groups)
    campaign = load_campaign(tmp_path)
    return od_report(summarise(campaign), str(tmp_path), campaign.truncated)


def named(report, name: str):
    """One criterion by exact name, or None."""
    for criterion in report.criteria:
        if criterion.name == name:
            return criterion
    return None


def healthy_run(run: int, scenario: str = "nominal", samples: int = 40):
    """One group of a consistent, accurate, un-faulted run."""
    return (
        run,
        scenario,
        [
            od_sample(
                i,
                nees=6.0,
                nis=3.0,
                err_ric_m=[1.0 if i % 2 else -1.0] * 3,
                sigma_ric_m=[1.0, 1.0, 1.0],
            )
            for i in range(samples)
        ],
    )


def test_a_healthy_campaign_passes(tmp_path: Path) -> None:
    """Consistent, accurate, ten runs: every criterion holds."""
    report = report_of(
        tmp_path, [[healthy_run(run)] for run in range(MIN_CAMPAIGN_RUNS)]
    )

    failures = [c.name for c in report.criteria if not c.passes]
    assert failures == []


def test_too_few_runs_fails_integrity_and_warns(tmp_path: Path) -> None:
    """A verdict read off a campaign that half ran points the wrong way."""
    report = report_of(tmp_path, [[healthy_run(run)] for run in range(3)])

    integrity = named(report, "Independent runs")
    assert integrity is not None
    assert not integrity.passes
    assert integrity.threshold == float(MIN_CAMPAIGN_RUNS)
    assert any("smoke test" in w for w in report.warnings)


def test_an_optimistic_filter_fails_the_upper_consistency_bound(tmp_path: Path) -> None:
    """A covariance smaller than the error is the unsafe direction and must fail.

    The two halves of the interval are separate criteria on purpose: a single
    "inside the interval" boolean would hide which way it went, and only one of
    the two ways is dangerous.
    """
    groups = [
        [
            (
                run,
                "nominal",
                [od_sample(i, nees=18.0, nis=3.0) for i in range(40)],
            )
        ]
        for run in range(MIN_CAMPAIGN_RUNS)
    ]
    report = report_of(tmp_path, groups)

    assert not named(report, "Campaign NEES not optimistic").passes
    assert named(report, "Campaign NEES not pessimistic").passes


def test_an_optimistic_covariance_fails_the_ensemble_check(tmp_path: Path) -> None:
    """The check NEES cannot make: truth says the spread is wider than claimed.

    Every run's error and covariance are internally consistent here — the NEES
    is exactly its expected value — while the covariance is a factor of three
    below the spread the ensemble of truth errors actually shows. This is the
    failure mode the ensemble criterion exists for, and it passes NEES.
    """
    groups = [
        [
            (
                run,
                "nominal",
                [
                    od_sample(
                        i,
                        nees=6.0,
                        err_ric_m=[3.0 if i % 2 else -3.0] * 3,
                        sigma_ric_m=[1.0, 1.0, 1.0],
                    )
                    for i in range(40)
                ],
            )
        ]
        for run in range(MIN_CAMPAIGN_RUNS)
    ]
    report = report_of(tmp_path, groups)

    assert named(report, "Campaign NEES not optimistic").passes
    radial = named(report, "Ensemble/reported sigma, radial")
    assert radial is not None
    assert not radial.passes
    assert radial.threshold == ENSEMBLE_RATIO_MAX
    assert radial.measured == 3.0


def test_a_gate_refusing_clean_fixes_fails(tmp_path: Path) -> None:
    """The threshold is the gate's own configuration, not taste.

    Refusing 20 % of clean fixes means the filter is under-weighting its own
    measurements, against a gate configured to refuse about 0.1 %.
    """
    groups = [
        [
            (
                run,
                "nominal",
                [od_sample(i) for i in range(32)]
                + [
                    od_sample(i, fix_accepted=0, refusal="innovation_gate")
                    for i in range(32, 40)
                ],
            )
        ]
        for run in range(MIN_CAMPAIGN_RUNS)
    ]
    report = report_of(tmp_path, groups)

    rejection = named(report, "Clean-fix rejection rate, nominal")
    assert rejection is not None
    assert not rejection.passes
    assert rejection.threshold == NOMINAL_REJECTION_CEILING


def test_a_geo_fix_caught_a_layer_in_fails_the_band_criterion(tmp_path: Path) -> None:
    """*Which* layer refused is the criterion, not that something did.

    The plausibility band is the only check on the seed path — a cold filter,
    or one whose solution the coast horizon just dropped, has no prior and so
    no innovation gate. A GEO-radius fix refused by the gate would have been
    accepted whole had it arrived one cycle earlier, and in any count that
    records only the refusal it looks identical to success.
    """
    groups = [
        [
            healthy_run(run),
            (
                run,
                "radius_jump",
                [
                    od_sample(
                        i,
                        regime="radius_jump",
                        fix_accepted=0,
                        refusal="innovation_gate",
                    )
                    for i in range(10)
                ],
            ),
        ]
        for run in range(MIN_CAMPAIGN_RUNS)
    ]
    report = report_of(tmp_path, groups)

    band = named(report, "Implausible fixes refused at the trust boundary")
    assert band is not None
    assert not band.passes
    assert band.measured == 0.0


def test_the_same_fix_caught_at_the_boundary_passes(tmp_path: Path) -> None:
    """The counterpart: refused on the band, which is the defence that generalises."""
    groups = [
        [
            healthy_run(run),
            (
                run,
                "radius_jump",
                [
                    od_sample(
                        i,
                        regime="radius_jump",
                        fix_accepted=0,
                        refusal="fix_implausible",
                    )
                    for i in range(10)
                ],
            ),
        ]
        for run in range(MIN_CAMPAIGN_RUNS)
    ]
    report = report_of(tmp_path, groups)

    assert named(report, "Implausible fixes refused at the trust boundary").passes


def test_the_spoof_drift_is_a_warning_and_never_a_criterion(tmp_path: Path) -> None:
    """No requirement bounds how far a slow spoof may walk the estimate.

    Reporting it as a criterion would mean inventing the threshold here and
    then passing against it. It is the number a spoofing requirement should be
    written from, so it is carried as a measurement.
    """
    groups = [
        [
            healthy_run(run),
            (
                run,
                "spoof_ramp",
                [od_sample(i, regime="spoof", pos_err_m=1873.0) for i in range(10)],
            ),
        ]
        for run in range(MIN_CAMPAIGN_RUNS)
    ]
    report = report_of(tmp_path, groups)

    assert any("1873" in w for w in report.warnings)
    assert not any("spoof" in c.name.lower() for c in report.criteria)


def test_a_still_flying_campaign_says_so(tmp_path: Path) -> None:
    """A half-written shard must not render identically to a finished campaign."""
    for run in range(MIN_CAMPAIGN_RUNS):
        write_od_shard(
            tmp_path / f"shard_{run}.jsonl",
            [healthy_run(run)],
            truncate_last=(run == 0),
        )
    campaign = load_campaign(tmp_path)
    report = od_report(summarise(campaign), str(tmp_path), campaign.truncated)

    assert any("partial record" in w for w in report.warnings)


def test_the_accuracy_numbers_are_provenance_not_criteria(tmp_path: Path) -> None:
    """Accuracy is reported and never gated: no requirement writes a number on it."""
    report = report_of(
        tmp_path, [[healthy_run(run)] for run in range(MIN_CAMPAIGN_RUNS)]
    )

    assert "Worst position error" in report.provenance
    assert "Steady-state position error" in report.provenance
    assert not any(
        "error" in c.name.lower() and "m" == c.units for c in report.criteria
    )
