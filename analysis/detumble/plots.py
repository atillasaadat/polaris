"""Figures that carry the campaign's verdict on their face.

Three, each answering one question the report makes a claim about:

``detumble_ensemble.png``
    Every run's rate history from engagement, with REQ-ACTL-001's fast-phase
    bound and the ``DetumbleExitRadps`` completion threshold drawn on the axes.
    This is the figure that shows the two timescales are different processes:
    a near-vertical collapse in the first minutes, then a slow geometry-driven
    unwind spread over orbits.

``detumble_time_to_exit.png``
    The empirical CDF of time-to-completion with the median, the tolerance bound
    and the proposed handover time marked — the distribution the handover number
    is read out of.

``detumble_drivers.png``
    Time-to-completion against the spin/field angle at the end of the fast phase
    and against the initial tip-off magnitude, side by side. The physics claim is
    that the first predicts the tail and the second does not; a scatter is how
    that claim is either shown or refuted.

Per the standing convention (``analysis/CLAUDE.md``) thresholds are drawn,
measurements are annotated where they were measured, verdicts appear as words as
well as colour, and everything lands in a caller-supplied directory under
``build-artifacts/`` — derived artifacts, never committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.common.plotting import (
    NEUTRAL_COLOR,
    annotate_measurement,
    plt,
    save,
    threshold_line,
    verdict_title,
)
from analysis.detumble.records import RunRecord
from analysis.detumble.report import FAST_PHASE_BOUND_DEG_S, FAST_PHASE_WINDOW_S
from analysis.detumble.statistics import DetumbleStatistics

__all__ = ["write_all"]

#: Default output directory for the figures and the rendered report.
DEFAULT_OUT = Path("build-artifacts/analysis/detumble")


def write_all(
    records: list[RunRecord],
    stats: DetumbleStatistics,
    report_text: str,
    out_dir: str | Path | None = None,
    *,
    passed: bool = True,
) -> list[Path]:
    """Write every figure and the rendered report.

    Parameters
    ----------
    records : list of RunRecord
        The campaign.
    stats : DetumbleStatistics
        Its summary.
    report_text : str
        Rendered report, written beside the figures.
    out_dir : str or pathlib.Path, optional
        Destination; defaults to :data:`DEFAULT_OUT`.
    passed : bool, optional
        Overall verdict, for the figure titles.

    Returns
    -------
    list of pathlib.Path
        The files written, in order.
    """
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    written = [
        _ensemble(records, stats, out, passed),
        _time_to_exit(stats, records, out, passed),
        _drivers(records, stats, out, passed),
    ]
    report_path = out / "detumble_mc_report.txt"
    report_path.write_text(report_text + "\n")
    written.append(report_path)
    return written


def _subtitle(stats: DetumbleStatistics) -> str:
    return (
        f"{stats.n_converged}/{stats.n_healthy} runs converged, "
        f"{stats.duration_s / stats.orbit_period_s:.1f}-orbit arc"
    )


def _ensemble(
    records: list[RunRecord], stats: DetumbleStatistics, out: Path, passed: bool
) -> Path:
    """Rate history of every run, on a log rate axis and an orbit time axis."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    exit_deg_s = _exit_threshold_deg_s(records)
    for r in records:
        if r.profile_t_s.size == 0:
            continue
        ax.plot(
            r.profile_t_s / stats.orbit_period_s,
            r.profile_rate_deg_s,
            lw=0.6,
            alpha=0.45,
            color=NEUTRAL_COLOR,
        )
    threshold_line(
        ax,
        FAST_PHASE_BOUND_DEG_S,
        f"REQ-ACTL-001 fast phase {FAST_PHASE_BOUND_DEG_S} deg/s",
    )
    threshold_line(
        ax, exit_deg_s, f"DetumbleExitRadps {exit_deg_s:.2f} deg/s", color="#8a5a00"
    )
    threshold_line(
        ax,
        FAST_PHASE_WINDOW_S / stats.orbit_period_s,
        f"{FAST_PHASE_WINDOW_S:.0f} s after engagement",
        orientation="v",
        color="#555555",
        ls=":",
    )
    annotate_measurement(
        ax,
        FAST_PHASE_WINDOW_S / stats.orbit_period_s,
        stats.worst_fast_phase_deg_s,
        f"worst {stats.worst_fast_phase_deg_s:.2f} deg/s",
        stats.worst_fast_phase_deg_s <= FAST_PHASE_BOUND_DEG_S,
    )
    ax.set_yscale("log")
    ax.set_xlabel("time from B-dot engagement [orbits]")
    ax.set_ylabel("body-rate magnitude [deg/s]")
    verdict_title(ax, "Detumble ensemble — leo-sso-500km", passed, _subtitle(stats))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    return save(fig, out / "detumble_ensemble.png")


def _time_to_exit(
    stats: DetumbleStatistics, records: list[RunRecord], out: Path, passed: bool
) -> Path:
    """Empirical CDF of time-to-completion, with the bound and the proposal."""
    times = np.sort(np.array([r.t_exit_s for r in records if r.converged], dtype=float))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    if times.size:
        cdf = np.arange(1, times.size + 1) / float(times.size)
        ax.step(times / stats.orbit_period_s, cdf, where="post", color=NEUTRAL_COLOR)
        annotate_measurement(
            ax,
            stats.median_s / stats.orbit_period_s,
            0.5,
            f"median {stats.median_s / stats.orbit_period_s:.2f} orbits",
        )
        annotate_measurement(
            ax,
            stats.worst_s / stats.orbit_period_s,
            1.0,
            f"worst {stats.worst_s / stats.orbit_period_s:.2f} orbits",
            offset=(-140, -14),
        )
    threshold_line(
        ax, stats.quantile, f"{stats.quantile:.0%} quantile", color="#555555", ls=":"
    )
    threshold_line(
        ax,
        stats.tolerance_bound_s / stats.orbit_period_s,
        f"{stats.quantile:.0%}/{stats.confidence:.0%} bound "
        f"{stats.tolerance_bound_s / stats.orbit_period_s:.2f} orbits",
        orientation="v",
    )
    threshold_line(
        ax,
        stats.handover_orbits,
        f"PROPOSED handover {stats.handover_orbits:.0f} orbits "
        f"({stats.handover_s:.0f} s)",
        orientation="v",
        color="#8a5a00",
    )
    ax.set_xlabel("time from B-dot engagement to DetumbleExitRadps [orbits]")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0.0, 1.05)
    verdict_title(
        ax, "Time to detumble completion — leo-sso-500km", passed, _subtitle(stats)
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    return save(fig, out / "detumble_time_to_exit.png")


def _drivers(
    records: list[RunRecord], stats: DetumbleStatistics, out: Path, passed: bool
) -> Path:
    """What predicts the tail: post-fast-phase geometry, not the tip-off."""
    converged = [r for r in records if r.converged]
    times = (
        np.array([r.t_exit_s for r in converged], dtype=float) / stats.orbit_period_s
    )
    panels = (
        (
            "spin/field angle after fast phase [deg]",
            np.array(
                [r.fast_phase_spin_field_angle_deg for r in converged], dtype=float
            ),
            "spin/field angle after fast phase [deg]",
        ),
        (
            "initial rate magnitude [deg/s]",
            np.array([r.rate_initial_deg_s for r in converged], dtype=float),
            "initial rate magnitude [deg/s]",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, (key, values, label) in zip(axes, panels, strict=True):
        ax.scatter(values, times, s=14, color=NEUTRAL_COLOR, alpha=0.7)
        rho = stats.correlations.get(key, float("nan"))
        ax.set_xlabel(label)
        ax.set_title(f"Spearman rho = {rho:+.2f}", fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("time to completion [orbits]")
    threshold_line(
        axes[0],
        stats.handover_orbits,
        f"PROPOSED handover {stats.handover_orbits:.0f} orbits",
        color="#8a5a00",
    )
    axes[0].legend(fontsize=8, loc="upper left")
    verdict_title(axes[0], "What sets the residual-spin tail", passed, _subtitle(stats))
    return save(fig, out / "detumble_drivers.png")


def _exit_threshold_deg_s(records: list[RunRecord]) -> float:
    """The completion threshold [deg/s] the runs were judged against.

    Read from the records, which carry the deployment's committed
    ``DetumbleExitRadps``. A figure drawing a hand-written 0.5 deg/s line beside
    data produced under some other committed value is a figure that lies
    quietly, so the number is never transcribed here.
    """
    return max((r.exit_threshold_deg_s for r in records), default=float("nan"))
