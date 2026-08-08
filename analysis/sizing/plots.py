"""Figures that show the sizing verdict on their face, plus the rendered report.

Per the standing convention (``analysis/CLAUDE.md``), none of these is a bare
curve: thresholds are drawn on the axes, measured values are annotated where
they were measured with the word PASS or FAIL, and every title carries the
configuration name and the verdict. Colour is never load-bearing on its own.

Three figures, each earning its place:

* :func:`envelope_figure` — **the headline.** The wheel array's momentum
  zonotope in 3D as a solid hull, with the L2 ellipsoid inscribed in it and the
  guaranteed-radius sphere, so the reader sees which surface is the binding one
  and how much the three differ.
* :func:`driver_figure` — its companion: every required-momentum driver against
  the usable envelope and the hardware radius, on one log axis. A bar above the
  usable-envelope line is a driver the design cannot hold. Deliberately a
  *second* figure rather than a second panel: the drivers on this class of
  vehicle are three orders of magnitude below the wheels' capability, and a
  shared linear axis renders every one of them as a dot at the origin.
* :func:`disturbance_figure` — the §5.3 budget as a labelled bar per term with
  its secular/cyclic split, and the totals beside it. The secular total is the
  number the rods must beat, so it is drawn against the rods' authority.
* :func:`magnetorquer_figure` — available magnetic torque against the secular
  disturbance it must beat, and the B-dot noise floor against the committed
  detumble exit threshold.

Units at the presentation boundary
----------------------------------
SI internally; the figures use N·m·s, µN·m and deg/s, which is where a reader's
intuition lives.

References
----------
Design doc §12 (analysis tools), §21.2 (generated artifacts).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from analysis.common.plotting import (
    NEUTRAL_COLOR,
    annotate_measurement,
    plt,
    save,
    threshold_line,
    verdict_color,
    verdict_title,
)
from analysis.control.vehicle import Vehicle
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.report import (
    SizingAnalysis,
    format_budget,
    format_derived,
    sizing_analysis,
    sizing_report,
)

#: Default output directory, relative to the repository root.
DEFAULT_OUTPUT_DIR = Path("build-artifacts/analysis/sizing")

#: Largest wheel count the 3D zonotope hull is drawn for. The vertex set is
#: :math:`2^N`, so this bounds the figure's cost; past it the ellipsoid and the
#: radii still draw and the hull is skipped with a note on the axes.
MAX_HULL_WHEELS = 12


def _prepare(out_dir: str | Path | None) -> Path:
    """Resolve and create the output directory."""
    path = Path(DEFAULT_OUTPUT_DIR if out_dir is None else out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _zonotope_vertices(axes: np.ndarray, capacity: float) -> np.ndarray:
    """Every :math:`W\\mathbf a` at a corner of the command box, shape ``(2^N, 3)``.

    The zonotope is the convex hull of these, so plotting them as a point cloud
    plus their hull outline shows the achievable set without a hull library.
    """
    signs = np.array(list(product((-1.0, 1.0), repeat=axes.shape[1])))
    return capacity * signs @ axes.T


def _unit_sphere(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit sphere mesh for the ellipsoid and guaranteed-radius surfaces."""
    u = np.linspace(0.0, 2.0 * np.pi, 2 * n)
    v = np.linspace(0.0, np.pi, n)
    return (
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
    )


def envelope_figure(
    analysis: SizingAnalysis, out_dir: str | Path | None = None
) -> Path:
    """The momentum zonotope in 3D, with the L2 ellipsoid inscribed in it.

    Three nested surfaces in body momentum space: the zonotope hull (what the
    L∞ allocator reaches), the guaranteed-radius sphere at
    :math:`r_{\\mathrm{in}}` (what is available in *every* direction, and what
    sizing uses), and the L2 ellipsoid (what a minimum-norm allocator reaches).
    The gap between the hull and the sphere is the anisotropy of the layout —
    the quantity a best-direction sizing number silently spends.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    momentum = analysis.wheels.momentum
    fig = plt.figure(figsize=(7.6, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    if momentum.n_actuators <= MAX_HULL_WHEELS:
        vertices = _zonotope_vertices(momentum.axes, momentum.capacity)
        hull = ConvexHull(vertices)
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=hull.simplices,
            color="#1f4e79",
            alpha=0.16,
            linewidth=0.2,
            edgecolor="#1f4e79",
        )
    else:  # pragma: no cover - no such vehicle in the repo
        ax.text2D(
            0.02,
            0.02,
            f"hull omitted: {momentum.n_actuators} wheels is "
            f"{2**momentum.n_actuators} vertices",
            transform=ax.transAxes,
            fontsize=7,
        )

    sx, sy, sz = _unit_sphere()
    semi = momentum.ellipsoid_semi_axes
    ax.plot_wireframe(
        semi[0] * sx,
        semi[1] * sy,
        semi[2] * sz,
        color="#7a5195",
        linewidth=0.35,
        rstride=4,
        cstride=4,
    )
    ax.plot_surface(
        momentum.inscribed * sx,
        momentum.inscribed * sy,
        momentum.inscribed * sz,
        alpha=0.22,
        color="#1a7f37",
        linewidth=0,
    )
    worst = momentum.worst_direction * momentum.inscribed
    ax.plot(
        [0.0, worst[0]], [0.0, worst[1]], [0.0, worst[2]], color="k", lw=1.6, zorder=10
    )
    ax.text(worst[0], worst[1], worst[2], "  weakest direction", fontsize=8)
    ax.set_xlabel("h_x [N.m.s]")
    ax.set_ylabel("h_y [N.m.s]")
    ax.set_zlabel("h_z [N.m.s]")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title(
        f"{analysis.vehicle.name} — wheel momentum envelope\n"
        f"zonotope (blue) reaches {momentum.circumscribed:.3g} at best, "
        f"guarantees {momentum.inscribed:.3g} (green sphere);\n"
        f"L2 ellipsoid (purple wireframe) guarantees "
        f"{momentum.ellipsoid_inscribed:.3g} N.m.s. Sizing uses the green radius.",
        fontsize=9,
    )
    return save(fig, _prepare(out_dir) / "momentum_envelope.png")


def driver_figure(analysis: SizingAnalysis, out_dir: str | Path | None = None) -> Path:
    """Required momentum per driver against the usable and hardware envelopes.

    Log axis, because the drivers and the wheels' capability differ by orders of
    magnitude on a vehicle whose ceiling is a linear-analysis boundary. A bar
    above the usable-envelope line is a driver the design cannot hold.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    wheels = analysis.wheels
    momentum = wheels.momentum
    drivers = [d for d in wheels.drivers if d.judged]
    passed = all(
        wheels.usable_momentum_nms >= analysis.assumptions.margin * d.required_nms
        for d in drivers
    )

    fig, ax2 = plt.subplots(figsize=(9.0, 5.4))
    positions = np.arange(len(drivers))
    required = [analysis.assumptions.margin * d.required_nms for d in drivers]
    verdicts = [wheels.usable_momentum_nms >= r for r in required]
    ax2.bar(
        positions,
        required,
        color=[verdict_color(v) for v in verdicts],
        label=f"required x {analysis.assumptions.margin:g} margin",
    )
    threshold_line(
        ax2,
        wheels.usable_momentum_nms,
        f"usable envelope {wheels.usable_momentum_nms:.3g} N.m.s",
        color="#1a7f37",
        ls="-",
    )
    threshold_line(
        ax2,
        momentum.inscribed,
        f"wheel hardware r_in {momentum.inscribed:.3g} N.m.s",
    )
    for x, (driver, value, ok) in enumerate(
        zip(drivers, required, verdicts, strict=True)
    ):
        annotate_measurement(
            ax2, x, value, f"{driver.required_nms:.2e} N.m.s", ok, offset=(4, 6)
        )
    ax2.set_yscale("log")
    ax2.set_xticks(positions)
    ax2.set_xticklabels([d.name.replace(" ", "\n", 1) for d in drivers], fontsize=8)
    ax2.set_ylabel("momentum [N.m.s], log scale")
    ax2.set_ylim(min(required) / 5.0, momentum.inscribed * 5.0)
    ax2.grid(True, axis="y", which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")
    verdict_title(
        ax2,
        f"{analysis.vehicle.name} — required momentum vs what the vehicle may use",
        passed,
        "a bar above the green line is a driver the certified envelope cannot hold",
    )
    return save(fig, _prepare(out_dir) / "momentum_drivers.png")


def disturbance_figure(
    analysis: SizingAnalysis, out_dir: str | Path | None = None
) -> Path:
    """The §5.3 budget per term, split secular/cyclic, against rod authority.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    budget = analysis.budget
    terms = budget.terms
    positions = np.arange(len(terms))
    secular = np.array([t.secular_nm for t in terms]) * 1e6
    cyclic = np.array([t.cyclic_nm for t in terms]) * 1e6

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax.bar(positions, secular, color="#b3261e", label="secular (sizes desaturation)")
    ax.bar(
        positions,
        cyclic,
        bottom=secular,
        color="#1f4e79",
        label="cyclic (sizes storage)",
    )
    for x, term in enumerate(terms):
        ax.text(
            x,
            (term.torque_nm) * 1e6,
            f"{term.torque_nm * 1e6:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([t.name.replace(" ", "\n") for t in terms], fontsize=8)
    ax.set_ylabel("disturbance torque [uN.m]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(
        f"{analysis.vehicle.name} — disturbance-torque budget\n"
        f"total {budget.total_nm * 1e6:.3g} uN.m = secular "
        f"{budget.secular_nm * 1e6:.3g} + cyclic {budget.cyclic_nm * 1e6:.3g}",
        fontsize=9,
    )

    margin = analysis.assumptions.margin
    required = margin * budget.secular_nm * 1e6
    available = analysis.mtq.average_torque_nm * 1e6
    ok = available >= required
    ax2.bar([0, 1], [required, available], color=[verdict_color(ok)] * 2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        [f"secular x {margin:g} margin", "rod average authority"], fontsize=9
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("torque [uN.m], log scale")
    ax2.grid(True, axis="y", which="both", alpha=0.3)
    annotate_measurement(ax2, 1, available, f"{available:.3g} uN.m", ok)
    verdict_title(
        ax2,
        "M1 desaturation authority",
        ok,
        "the rods must beat the secular torque, or the wheels saturate\n"
        "whatever their size",
    )
    return save(fig, _prepare(out_dir) / "disturbance_budget.png")


def magnetorquer_figure(
    analysis: SizingAnalysis, out_dir: str | Path | None = None
) -> Path:
    """Rod detumble authority and the B-dot noise floor against the thresholds.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    vehicle = analysis.vehicle
    mtq = analysis.mtq
    floor = mtq.noise_floor
    margin = analysis.assumptions.margin

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    removable = mtq.removable_momentum_nms
    required = margin * mtq.tipoff_momentum_nms
    ok = removable >= required
    ax.bar([0, 1], [required, removable], color=[verdict_color(ok)] * 2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"tip-off momentum x {margin:g}", "removable in the budget"], fontsize=9
    )
    ax.set_yscale("log")
    ax.set_ylabel("momentum [N.m.s], log scale")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    annotate_measurement(ax, 1, removable, f"{removable:.3g} N.m.s", ok)
    verdict_title(
        ax,
        f"{vehicle.name} — M2 detumble authority",
        ok,
        f"implied fast-phase duration {mtq.implied_detumble_s:.0f} s of a "
        f"{mtq.detumble_budget_s:.0f} s budget",
    )

    rates = np.array(
        [
            np.degrees(vehicle.detumble_exit_radps),
            np.degrees(floor.rate_mean_radps),
            np.degrees(floor.rate_worst_radps),
        ]
    )
    exit_ok = vehicle.detumble_exit_radps >= floor.rate_worst_radps
    # Only the first bar is a judged quantity; the two floors are measurements
    # with no threshold of their own, so they are drawn neutral rather than in a
    # verdict colour that would read as a failure of the magnetometer.
    ax2.bar(
        np.arange(3),
        rates,
        color=[verdict_color(exit_ok), NEUTRAL_COLOR, NEUTRAL_COLOR],
    )
    threshold_line(
        ax2,
        rates[2],
        f"noise floor at |B|_min ({rates[2]:.2f} deg/s)",
    )
    for x, value in enumerate(rates):
        ax2.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(
        ["DetumbleExitRadps", "floor at |B|_mean", "floor at |B|_min"], fontsize=8
    )
    ax2.set_ylabel("body rate [deg/s]")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    verdict_title(
        ax2,
        "M3 B-dot measurement floor",
        exit_ok,
        "an exit threshold under the floor declares detumble complete on noise",
    )
    return save(fig, _prepare(out_dir) / "magnetorquer_sizing.png")


def write_all(
    vehicle: Vehicle,
    out_dir: str | Path | None = None,
    config_path: str | Path = "",
    assumptions: SizingAssumptions | None = None,
) -> list[Path]:
    """Generate every figure **and** the rendered text report.

    The report file carries the criteria table, the disturbance budget and the
    derived-parameter justifications, so it stands alone as the record.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.
    config_path : str or pathlib.Path, optional
        The config the vehicle came from, recorded in the report's provenance.
    assumptions : SizingAssumptions, optional
        The assumptions in force.

    Returns
    -------
    list of pathlib.Path
        The files written, report last.
    """
    directory = _prepare(out_dir)
    analysis = sizing_analysis(vehicle, assumptions)
    figures = [
        envelope_figure(analysis, directory),
        driver_figure(analysis, directory),
        disturbance_figure(analysis, directory),
        magnetorquer_figure(analysis, directory),
    ]
    report = sizing_report(vehicle, config_path, analysis.assumptions, analysis)
    target = directory / "sizing_report.txt"
    target.write_text(
        report.format_text()
        + "\n\n"
        + format_budget(analysis)
        + "\n\n"
        + format_derived(analysis)
        + "\n"
    )
    return [*figures, target]
