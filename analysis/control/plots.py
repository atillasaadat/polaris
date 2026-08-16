"""Figures that show the verdict on their face, plus the rendered text report.

Per the standing convention (``analysis/CLAUDE.md``), none of these is a bare
curve the reader has to interpret: every requirement threshold is drawn on the
axes, every measured margin is annotated at the frequency it was measured with
its value and the word PASS or FAIL, and each title carries the configuration
name and the overall verdict. Colour is never load-bearing on its own.

Every function takes the output directory from the caller and returns the paths
written. The default is ``build-artifacts/analysis/control`` — the repo's
existing generated-output location, already ignored by git. Figures and the
text report are *derived* artifacts and are never committed; the numbers that
matter live in the structured
:class:`analysis.common.report.AnalysisReport` the test suite asserts on.

Units at the presentation boundary
----------------------------------
SI internally, friendly units on the axes: frequency in rad/s on a log scale,
magnitude in dB, phase and separations in degrees.

References
----------
Design doc §12 (analysis tools), §21.2 (generated artifacts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.common.plotting import (
    annotate_measurement,
    plt,
    save,
    threshold_line,
    verdict_color,
    verdict_title,
)
from analysis.control.controllability import (
    wheel_controllability,
    wheel_failure_subsets,
)
from analysis.control.margins import (
    MIN_GAIN_MARGIN_DB,
    MIN_PHASE_MARGIN_DEG,
    frequency_grid,
)
from analysis.control.observability import (
    two_vector_observability,
    vector_geometry_sweep,
)
from analysis.control.plant import AXES, discrete_open_loop, open_loop
from analysis.control.report import control_analysis_report, margin_report
from analysis.control.vehicle import Vehicle

#: Default output directory, relative to the repository root.
DEFAULT_OUTPUT_DIR = Path("build-artifacts/analysis/control")


def _prepare(out_dir: str | Path | None) -> Path:
    """Resolve and create the output directory."""
    path = Path(DEFAULT_OUTPUT_DIR if out_dir is None else out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def bode_figure(vehicle: Vehicle, out_dir: str | Path | None = None) -> Path:
    """Open-loop Bode plot per axis, with the margins marked where they occur.

    Drawn per axis rather than all three on one pair of axes, because the point
    of the figure is the annotation: the gain crossover carries its phase margin
    against the −180° + PM requirement bound, and the low-frequency −180°
    crossing carries the downward gain margin that makes this loop
    conditionally stable. The continuous curve is drawn behind the sampled one
    so the cost of the zero-order hold is visible rather than asserted.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    report = margin_report(vehicle)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(14.0, 7.0))
    for i, result in enumerate(report.axes):
        ax_mag, ax_phase = axs[0, i], axs[1, i]
        sampled = discrete_open_loop(vehicle, i)
        omega = frequency_grid(sampled, points=1500)
        response = sampled.response(omega)
        mag_db = 20.0 * np.log10(np.abs(response))
        phase = np.degrees(np.unwrap(np.angle(response)))

        continuous = open_loop(vehicle, i)
        cont_resp = continuous.response(omega)
        ax_mag.semilogx(
            omega,
            20.0 * np.log10(np.abs(cont_resp)),
            "--",
            color="grey",
            lw=0.9,
            label="continuous",
        )
        ax_phase.semilogx(
            omega,
            np.degrees(np.unwrap(np.angle(cont_resp))),
            "--",
            color="grey",
            lw=0.9,
        )
        ax_mag.semilogx(omega, mag_db, color="#1f4e79", label="sampled 10 Hz")
        ax_phase.semilogx(omega, phase, color="#1f4e79")

        threshold_line(ax_mag, 0.0, "0 dB (crossover)", color="k", ls="-", lw=0.6)
        # The downward gain margin, at the low-frequency -180 crossing.
        if result.phase_crossover_rad_s:
            w180 = result.phase_crossover_rad_s[0]
            annotate_measurement(
                ax_mag,
                w180,
                float(20.0 * np.log10(np.abs(sampled.response(w180)))),
                f"GM {result.gain_margin_down_db:.1f} dB down (req {MIN_GAIN_MARGIN_DB:g})",
                result.gain_margin_db >= MIN_GAIN_MARGIN_DB,
                offset=(8, -18),
            )
        threshold_line(ax_phase, -180.0, "-180 deg", color="k", ls=":", lw=0.8)
        threshold_line(
            ax_phase,
            -180.0 + MIN_PHASE_MARGIN_DEG,
            f"PM req ({MIN_PHASE_MARGIN_DEG:g} deg)",
        )
        annotate_measurement(
            ax_phase,
            result.gain_crossover_rad_s,
            -180.0 + result.phase_margin_deg,
            f"PM {result.phase_margin_deg:.1f} deg",
            result.phase_margin_deg >= MIN_PHASE_MARGIN_DEG,
        )

        verdict_title(
            ax_mag,
            f"{vehicle.name} — axis {AXES[i]}",
            result.passes,
            f"wc = {result.gain_crossover_rad_s:.3f} rad/s",
        )
        ax_mag.set_ylim(-80.0, 80.0)
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend(fontsize=7, loc="upper right")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.set_xlabel("frequency [rad/s]")
        ax_phase.legend(fontsize=7, loc="lower right")
        if i == 0:
            ax_mag.set_ylabel("magnitude [dB]")
            ax_phase.set_ylabel("phase [deg]")
    fig.suptitle(
        f"Open-loop response and REQ-ACTL-006 margins — "
        f"{'PASS' if report.passes else 'FAIL'}",
        color=verdict_color(report.passes),
    )
    return save(fig, _prepare(out_dir) / "bode_open_loop.png")


def nyquist_figure(vehicle: Vehicle, out_dir: str | Path | None = None) -> Path:
    """Nyquist plot of the sampled loop with the disk-margin exclusion region.

    Two things the picture is for. The locus crosses the negative real axis
    **left of** −1 at low frequency, which is what makes the loop conditionally
    stable and its gain margin a *downward* one. And the shaded disc is the
    region the symmetric disk margin excludes: the locus staying clear of it is
    the guarantee against *simultaneous* gain and phase error that the two
    classical margins separately do not give.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    report = margin_report(vehicle)
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    for i, result in enumerate(report.axes):
        loop = discrete_open_loop(vehicle, i)
        omega = frequency_grid(loop, points=4000)
        response = loop.response(omega)
        ax.plot(response.real, response.imag, label=f"axis {AXES[i]}", lw=1.2)
        ax.plot(response.real, -response.imag, lw=0.6, alpha=0.35, color="grey")

    # Symmetric disk margin: the excluded disc is centred on
    # -(2+a^2/... ) - use the standard skew-0 disc through the gain-margin
    # endpoints -(2+a)/(2-a) and -(2-a)/(2+a) on the real axis.
    worst = min(report.axes, key=lambda a: a.disk_margin)
    alpha = worst.disk_margin
    lo, hi = -(2.0 + alpha) / (2.0 - alpha), -(2.0 - alpha) / (2.0 + alpha)
    center, radius = 0.5 * (lo + hi), 0.5 * (hi - lo)
    ax.add_patch(
        plt.Circle(
            (center, 0.0),
            radius,
            color=verdict_color(report.passes),
            alpha=0.12,
            zorder=0,
            label=f"disk-margin exclusion (alpha = {alpha:.2f})",
        )
    )
    ax.plot([-1.0], [0.0], "r+", markersize=13, zorder=5, label="critical point -1")
    unit = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(unit), np.sin(unit), ":", color="grey", lw=0.8)
    ax.set_xlim(-8.0, 3.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Re L")
    ax.set_ylabel("Im L")
    ax.legend(fontsize=8, loc="upper left")
    verdict_title(
        ax,
        f"{vehicle.name} — Nyquist, sampled loop",
        report.passes,
        f"worst disk margin {alpha:.2f}: {worst.disk_gain_margin_db:.1f} dB and "
        f"{worst.disk_phase_margin_deg:.1f} deg simultaneously",
    )
    return save(fig, _prepare(out_dir) / "nyquist_open_loop.png")


def margin_figure(vehicle: Vehicle, out_dir: str | Path | None = None) -> Path:
    """Per-axis margin bars against the REQ-ACTL-006 thresholds.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    report = margin_report(vehicle)
    positions = np.arange(3)
    fig, (ax_gm, ax_pm) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    panels = (
        (ax_gm, "gain margin [dB]", MIN_GAIN_MARGIN_DB, "gain_margin_db", "dB"),
        (ax_pm, "phase margin [deg]", MIN_PHASE_MARGIN_DEG, "phase_margin_deg", "deg"),
    )
    for ax, ylabel, threshold, attr, units in panels:
        values = [getattr(a, attr) for a in report.axes]
        colors = [verdict_color(v >= threshold) for v in values]
        ax.bar(positions, values, color=colors)
        threshold_line(ax, threshold, f"requirement {threshold:g} {units}")
        for x, value, result in zip(positions, values, report.axes, strict=True):
            ax.text(
                x,
                value,
                f"{value:.1f}\n{'PASS' if value >= threshold else 'FAIL'}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=verdict_color(value >= threshold),
            )
            del result
        ax.set_xticks(positions)
        ax.set_xticklabels([f"axis {a.axis}" for a in report.axes])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, max(max(values), threshold) * 1.35)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"{vehicle.name} — REQ-ACTL-006 stability margins, sampled loop at "
        f"{vehicle.control_period_s:g} s — {'PASS' if report.passes else 'FAIL'}",
        color=verdict_color(report.passes),
    )
    return save(fig, _prepare(out_dir) / "margins.png")


def controllability_figure(vehicle: Vehicle, out_dir: str | Path | None = None) -> Path:
    """Wheel-array conditioning per failure case against the flight allocator's gate.

    The acceptance floor drawn on this figure is ``AllocMinConditioning`` read
    from the vehicle's own config, not a number chosen for the plot: a subset
    below it is one the flight allocator would refuse to allocate on.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    cases = {"all 4": wheel_controllability(vehicle)}
    cases.update(
        {
            "-".join(str(w) for w in wheels): result
            for wheels, result in wheel_failure_subsets(vehicle).items()
        }
    )
    floor = vehicle.alloc_min_conditioning
    labels = list(cases)
    values = [c.input_conditioning for c in cases.values()]
    ranks = [c.kalman_rank for c in cases.values()]
    passed = [v >= floor and r == 6 for v, r in zip(values, ranks, strict=True)]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(np.arange(len(labels)), values, color=[verdict_color(p) for p in passed])
    threshold_line(ax, floor, f"AllocMinConditioning ({floor:g})")
    for x, (value, rank, ok) in enumerate(zip(values, ranks, passed, strict=True)):
        ax.text(
            x,
            value,
            f"{value:.3f}\nrank {rank}/6\n{'PASS' if ok else 'FAIL'}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=verdict_color(ok),
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([f"wheels {label}" for label in labels])
    ax.set_ylabel(r"$\lambda_{min}/\lambda_{max}$ of $AA^{T}$ [-]")
    ax.set_ylim(0.0, max(max(values), floor) * 1.45)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    verdict_title(
        ax,
        f"{vehicle.name} — REQ-ACTL-007 wheel-array controllability",
        all(passed),
        "one bar per single-wheel-failure case; floor is the flight allocator's gate",
    )
    return save(fig, _prepare(out_dir) / "controllability.png")


def observability_figure(vehicle: Vehicle, out_dir: str | Path | None = None) -> Path:
    """Attitude information ratio against sun/field separation.

    Marks the coarse chain's ``MinSinAngle`` geometry gate, so the figure shows
    what the flight software refuses rather than only what it can do.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    separations = np.linspace(np.deg2rad(0.5), np.pi / 2.0, 400)
    ratios = vector_geometry_sweep(vehicle, separations)
    gate_rad = float(np.arcsin(vehicle.sensors.min_sin_angle))
    gate_ratio = float(vector_geometry_sweep(vehicle, np.array([gate_rad]))[0])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogy(np.degrees(separations), ratios, color="#1f4e79")
    threshold_line(
        ax,
        np.degrees(gate_rad),
        f"MinSinAngle gate ({np.degrees(gate_rad):.1f} deg)",
        orientation="v",
    )
    # No verdict word on these two: the information ratio has no threshold to be
    # judged against — REQ-ACTL-008 is written on the *rank*, and the flight
    # MinSinAngle gate is a design choice about usable accuracy rather than a
    # requirement bound. A PASS that cannot fail is not a verdict.
    annotate_measurement(
        ax, np.degrees(gate_rad), gate_ratio, f"ratio {gate_ratio:.2e} at the gate"
    )
    annotate_measurement(
        ax, 90.0, float(ratios[-1]), f"ratio {ratios[-1]:.3f} at 90 deg"
    )
    ax.set_xlabel("sun / field separation [deg]")
    ax.set_ylabel(r"$\lambda_{min}/\lambda_{max}$ of the information matrix [-]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    # The title's verdict *is* falsifiable — REQ-ACTL-008 is written on the rank
    # — so it is measured rather than asserted.
    nominal = two_vector_observability(vehicle)
    verdict_title(
        ax,
        f"{vehicle.name} — REQ-ACTL-008 two-vector observability",
        nominal.observable,
        f"rank {nominal.rank} of {nominal.n_states} above 0 deg separation; "
        "what degrades toward parallel is the conditioning",
    )
    return save(fig, _prepare(out_dir) / "observability_geometry.png")


def write_all(
    vehicle: Vehicle,
    out_dir: str | Path | None = None,
    config_path: str | Path = "",
) -> list[Path]:
    """Generate every figure **and** the rendered text report.

    The convention is both, always: the figures for the reader, the text report
    for the record. The structured report the tests assert on is
    :func:`analysis.control.report.control_analysis_report`; this writes its
    rendering.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.
    config_path : str or pathlib.Path, optional
        The config the vehicle came from, recorded in the report's provenance.

    Returns
    -------
    list of pathlib.Path
        The files written, report last.
    """
    directory = _prepare(out_dir)
    figures = [
        bode_figure(vehicle, directory),
        nyquist_figure(vehicle, directory),
        margin_figure(vehicle, directory),
        controllability_figure(vehicle, directory),
        observability_figure(vehicle, directory),
    ]
    report = control_analysis_report(vehicle, config_path)
    return [*figures, report.write_text(directory / "control_analysis_report.txt")]
