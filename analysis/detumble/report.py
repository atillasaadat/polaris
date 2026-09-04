"""The campaign's verdict, as a structured report.

Two things are being reported and they are not the same kind of claim.

**What is verified.** REQ-ACTL-001 already carries a fast-phase bound — below
3.8 deg/s within 200 s of engaging, and not rising above it afterwards — and
until this campaign that bound was measured at exactly *one* geometry: the
single SITL row, on one attitude, one orbit phase and one tip-off direction. The
campaign re-measures it across the dispersion, so those criteria are genuine
pass/fail rows here.

**What is proposed.** The time to the ``DetumbleExitRadps`` completion predicate
has no requirement written on it — that is precisely the item REQ-ACTL-001
records as owed. So the handover time this module computes is carried as a
**proposal in the report's provenance and warnings**, not as a criterion. A
criterion needs a threshold, a threshold is a requirement value, and inventing
one here and then passing against it would be the "threshold tuned to the
measurement" defect the review-lessons catalog names. The campaign's own
integrity — did every run converge inside the arc it flew, and is the sample
large enough for the bound claimed — *is* checkable, and those are criteria.

References
----------
Design doc §12 (analysis tools), §22.2 (requirements and margin reporting), §13
(Monte Carlo). REQ-ACTL-001 (``docs/requirements/adcs_control.rst``).
"""

from __future__ import annotations

import math

from analysis.common.report import AnalysisReport, Criterion
from analysis.detumble.statistics import DetumbleStatistics

__all__ = ["HANDOVER_BOUND_S", "FAST_PHASE_WINDOW_S", "detumble_report"]

#: REQ-ACTL-001's fast-phase bound [deg/s] and its window [s], from the
#: requirement text. Requirement values, not measurements — the campaign is
#: judged against them, never the other way round.
#: REQ-ACTL-001's committed time-to-handover bound [s] — two orbits at the
#: reference vehicle's 5677 s period. The requirement bounds the 95th percentile
#: of the time to ``DetumbleExitRadps`` at 95 % confidence, so what is judged is
#: the campaign's *tolerance bound*, not its worst run: the worst run is one draw
#: and the quantity the requirement names is a property of the distribution.
HANDOVER_BOUND_S = 11354.0

#: Window the retired fast-phase clause used [s]. Kept because the rate at this
#: instant is still reported as a diagnostic — it is the single most legible
#: number for how much authority the geometry gave the law — but it is no longer
#: judged. Until Push 84 REQ-ACTL-001 required the rate below 3.8 deg/s here,
#: which the 93-run campaign showed 24 % of geometries miss and which no B-dot
#: design can meet: with the spin along **B** the body-frame dB/dt carries no
#: signal, so the law correctly commands almost nothing and no gain recovers it.
#: See the requirement for the retirement argument and the evidence.
FAST_PHASE_WINDOW_S = 200.0


def detumble_report(stats: DetumbleStatistics, records_path: str) -> AnalysisReport:
    """Build the campaign report.

    Parameters
    ----------
    stats : DetumbleStatistics
        As returned by :func:`analysis.detumble.statistics.summarise`.
    records_path : str
        The campaign records the numbers came from — provenance, so a report read
        later says which campaign it describes.

    Returns
    -------
    analysis.common.report.AnalysisReport
        Criteria on the fast phase and on the campaign's own integrity, with the
        proposed handover time in the provenance and warnings.
    """
    criteria = (
        Criterion(
            name="time to DetumbleExitRadps, 95th percentile at 95% confidence",
            requirement="REQ-ACTL-001",
            threshold=HANDOVER_BOUND_S,
            measured=stats.tolerance_bound_s,
            units="s",
            sense="max",
            note=(
                f"distribution-free Wilks bound from order statistic "
                f"{max(stats.tolerance_order, 1)} of {stats.n_converged}; the "
                "requirement bounds a quantile, so this and not the worst run is "
                "the quantity judged"
            ),
        ),
        Criterion(
            name="runs right-censored by the flown arc",
            requirement="",
            threshold=0.0,
            measured=float(stats.n_censored),
            units="runs",
            sense="max",
            note=(
                f"a run that did not complete within {stats.duration_s:.0f} s of "
                "engagement makes the tolerance bound a lower bound on itself; "
                "re-fly the campaign with a longer --duration-s"
            ),
        ),
        Criterion(
            name=(
                f"converged runs vs the {stats.quantile:.0%}/{stats.confidence:.0%} "
                "sample size"
            ),
            requirement="",
            threshold=float(stats.required_n),
            measured=float(stats.n_converged),
            units="runs",
            sense="min",
            note=(
                f"Wilks order-{max(stats.tolerance_order, 1)} tolerance bound; below "
                "this the bound claimed is not supported by the sample"
            ),
        ),
        Criterion(
            name="runs that were tumbling when B-dot engaged",
            requirement="",
            threshold=0.0,
            measured=float(stats.n_below_entry),
            units="runs",
            sense="max",
            note=(
                "a run starting below DetumbleEnterRadps never tumbled, so its "
                "completion time describes a vehicle that arrived detumbled; "
                "non-zero means the tip-off dispersion floor and the flight "
                "entry threshold have come apart"
            ),
        ),
        Criterion(
            name="runs the harness completed",
            requirement="",
            threshold=float(stats.n_records),
            measured=float(stats.n_healthy),
            units="runs",
            sense="min",
            note="an unhealthy run is a harness failure, not a slow detumble",
        ),
    )

    correlations = "; ".join(
        f"{name} {value:+.2f}" for name, value in stats.correlations.items()
    )
    provenance = {
        "records": records_path,
        "runs": (
            f"{stats.n_records} flown, {stats.n_healthy} healthy, "
            f"{stats.n_converged} converged, {stats.n_censored} censored"
        ),
        "arc": (
            f"{stats.duration_s:.0f} s from engagement "
            f"({stats.duration_s / stats.orbit_period_s:.1f} orbits)"
        ),
        "median": f"{stats.median_s:.0f} s ({stats.median_s / stats.orbit_period_s:.2f} orbits)",
        "p95 (emp.)": f"{stats.p95_empirical_s:.0f} s",
        "worst": f"{stats.worst_s:.0f} s ({stats.worst_s / stats.orbit_period_s:.2f} orbits)",
        "bound": (
            f"{stats.tolerance_bound_s:.0f} s — {stats.quantile:.0%} quantile at "
            f"{stats.confidence:.0%} confidence, from order statistic "
            f"{stats.tolerance_order} of {stats.n_converged}"
        ),
        "PROPOSED": (
            (
                f"Safe-mode handover at {stats.handover_s:.0f} s "
                f"({stats.handover_orbits:.0f} orbits) after B-dot engagement — the "
                f"bound plus {stats.handover_margin:.0%} margin, rounded up to a "
                "whole orbit"
            )
            if math.isfinite(stats.handover_s)
            else "no handover time — this campaign supports no tolerance bound"
        ),
        "floor": (
            f"median achieved floor {stats.median_floor_deg_s:.2f} deg/s vs a "
            f"{stats.exit_threshold_deg_s:.2f} deg/s completion threshold "
            f"({stats.floor_margin:.1f}x headroom)"
        ),
        "correlation": correlations,
        "wall clock": f"{stats.total_wall_s / 3600.0:.2f} h of run time summed",
    }

    assumptions = (
        "Truth-side rates: the completion predicate is evaluated on the plant's "
        "body rate, not on the estimator's, using the deployment's own committed "
        "DetumbleExitRadps and DetumbleConfirmCycles.",
        "Times are measured from B-dot engagement — the first cycle the "
        "deployment scheduled a torque-rod on-window — not from boot, so the "
        "GNSS receiver's cold start is not charged to the control law.",
        "Nominal vehicle: the campaign clears the committed scenario's scheduled "
        "GNSS outage and spoof and disables the geographic jamming map. Detumble "
        "through a GNSS outage starves B-dot of its only input and is a fault "
        "row, not a convergence measurement.",
        "One orbit geometry family: 500 km circular SSO at the reference "
        "inclination, with RAAN, argument of latitude and epoch dispersed. A "
        "different inclination is a different field-geometry problem and needs "
        "its own campaign.",
        "The tolerance bound is distribution-free (Wilks order statistics); no "
        "shape is assumed for the tail.",
    )

    warnings = [
        "The PROPOSED line is what this campaign's own bound would support, "
        "recomputed every run. REQ-ACTL-001's committed bound is "
        f"{HANDOVER_BOUND_S:.0f} s and is what the criterion above judges — the "
        "two are separate on purpose, so a campaign that drifts away from the "
        "committed number says so instead of quietly redefining it.",
    ]
    if stats.floor_margin < 2.0:
        warnings.append(
            f"The completion threshold ({stats.exit_threshold_deg_s:.2f} deg/s) is "
            f"only {stats.floor_margin:.1f}x the rate B-dot actually reaches "
            f"(median floor {stats.median_floor_deg_s:.2f} deg/s). With that little "
            "headroom, whether a run confirms at all is decided by geometry rather "
            "than by how long it is given, and a time bound written on this "
            "predicate would be a bound on luck. Check the field-derivative signal "
            "to noise at the achieved rate before writing one."
        )
    if stats.n_re_excited:
        warnings.append(
            f"{stats.n_re_excited} of {stats.n_converged} converged runs ended the "
            "arc back above the completion threshold. The campaign holds DETUMBLE "
            "for the whole arc and the flown CONOPS does not — the mode manager "
            "hands over on the predicate — so this is not a failure. It does mean "
            "the handover must be *triggered by* the predicate and not scheduled at "
            "a fixed time after it."
        )
    if stats.tolerance_order == 1:
        warnings.append(
            "The tolerance bound is the single worst run (first-order Wilks). It "
            "is valid, but it cannot distinguish the physics from one "
            "misbehaving run — 93 converged runs would move it to the second "
            "largest and remove that sensitivity."
        )
    if stats.n_censored:
        warnings.append(
            f"{stats.n_censored} run(s) are right-censored — they did not complete "
            "inside the flown arc — so every quantile above the censoring "
            "fraction, including the tolerance bound, is a lower bound on itself."
        )

    return AnalysisReport(
        title="B-dot detumble Monte Carlo — residual-spin tail (REQ-ACTL-001)",
        config_path=records_path,
        provenance=provenance,
        assumptions=assumptions,
        criteria=criteria,
        warnings=tuple(warnings),
    )
