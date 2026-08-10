"""Figures that show the sizing verdict on their face, plus the rendered report.

Per the standing convention (``analysis/CLAUDE.md``), none of these is a bare
curve: thresholds are drawn on the axes, measured values are annotated where
they were measured with the word PASS or FAIL, and every title carries the
configuration name and the verdict. Colour is never load-bearing on its own.

Four figures, each earning its place, plus :func:`write_text_report` — the
plain-text rendering, which lives here beside them but is **not** one of them:
it is the record, so a run that skips the figures still writes it.

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
SI internally; the figures pick an SI prefix per axis from the magnitudes they
carry (:func:`analysis.sizing.mathfmt.unit_scale`) and state rates in deg/s,
which is where a reader's intuition lives, set with the unit symbols rather than the console's dotted
convention.

The same palette as the page
----------------------------
These PNGs are embedded in the HTML report and go into design reviews, so they
share its rules: titles in title case, axis labels capitalised and carrying their
units, symbols set as mathtext (``$r_{in}$``, ``$|B|_{min}$``), annotated values
in a mono face so a column of them aligns, a recessive hairline grid, and the
**status colours reserved for verdicts** — a category or a threshold line is
drawn in the series or structure colour, never in the PASS green or FAIL red,
and every verdict is a word as well as a colour.

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
    annotate_measurement,
    plt,
    save,
    threshold_line,
    verdict_color,
    verdict_title,
)
from analysis.control.vehicle import Vehicle
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.mathfmt import unit_html, unit_scale
from analysis.sizing.interactive import (
    ELLIPSOID_COLOR,
    GRID_COLOR,
    SERIES_1,
    SERIES_2,
    STRUCTURE_COLOR,
)
from analysis.sizing.report import (
    SizingAnalysis,
    format_budget,
    format_derived,
    sizing_analysis,
    sizing_report,
)

#: Default output directory, relative to the repository root.
DEFAULT_OUTPUT_DIR = Path("build-artifacts/analysis/sizing")

#: The mono face numbers are annotated in, so a column of values on a figure
#: aligns the way it does in the page's tables. Bundled with matplotlib, so this
#: adds no font dependency.
MONO = "DejaVu Sans Mono"

#: Grid weight. A grid is a reading aid behind the data, never a mark competing
#: with it, so it is a hairline in the page's rule colour rather than a
#: half-opaque copy of the ink.
GRID_WIDTH = 0.6

#: Line weight for anything that carries data or a threshold. Matches the 2 px
#: the interactive figures use, so the two renderings look like one set.
LINE_WIDTH = 2.0


def _upper_first(text: str) -> str:
    """Capitalise the first character and leave every other one alone.

    ``str.capitalize`` would lower-case the rest, which turns "post-B-dot
    handover" into "post-b-dot handover" and renames the algorithm.
    """
    return text[:1].upper() + text[1:]


def _grid(ax, which: str = "major", axis: str = "y") -> None:
    """A recessive hairline grid, and no frame competing with it."""
    ax.grid(True, axis=axis, which=which, color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)


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
    # One prefix for the whole figure: three nested surfaces compared by eye
    # have to be compared in one unit.
    scale = unit_scale(
        "N.m.s",
        [momentum.circumscribed, momentum.inscribed, momentum.ellipsoid_inscribed],
    )
    units = unit_html(scale.units)
    fig = plt.figure(figsize=(7.6, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    if momentum.n_actuators <= MAX_HULL_WHEELS:
        vertices = scale.factor * _zonotope_vertices(momentum.axes, momentum.capacity)
        hull = ConvexHull(vertices)
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=hull.simplices,
            color=STRUCTURE_COLOR,
            alpha=0.16,
            linewidth=0.2,
            edgecolor=STRUCTURE_COLOR,
        )
    else:  # pragma: no cover - no such vehicle in the repo
        ax.text2D(
            0.02,
            0.02,
            f"Hull omitted: {momentum.n_actuators} wheels is "
            f"{2**momentum.n_actuators} vertices",
            transform=ax.transAxes,
            fontsize=7,
        )

    sx, sy, sz = _unit_sphere()
    semi = scale.factor * momentum.ellipsoid_semi_axes
    ax.plot_wireframe(
        semi[0] * sx,
        semi[1] * sy,
        semi[2] * sz,
        color=ELLIPSOID_COLOR,
        linewidth=0.35,
        rstride=4,
        cstride=4,
    )
    inscribed = scale.value(momentum.inscribed)
    ax.plot_surface(
        inscribed * sx,
        inscribed * sy,
        inscribed * sz,
        alpha=0.22,
        color=SERIES_1,
        linewidth=0,
    )
    worst = momentum.worst_direction * inscribed
    ax.plot(
        [0.0, worst[0]],
        [0.0, worst[1]],
        [0.0, worst[2]],
        color="k",
        lw=LINE_WIDTH,
        zorder=10,
    )
    ax.text(worst[0], worst[1], worst[2], "  Weakest direction", fontsize=8)
    ax.set_xlabel(f"$h_x$ [{units}]")
    ax.set_ylabel(f"$h_y$ [{units}]")
    ax.set_zlabel(f"$h_z$ [{units}]")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title(
        f"Wheel Momentum Envelope, {analysis.vehicle.name}\n"
        f"Zonotope (slate hull) reaches {scale.text(momentum.circumscribed, 3)} at "
        f"best and guarantees $r_{{in}}$ = {scale.text(momentum.inscribed, 3)} "
        f"(blue sphere);\nthe L2 ellipsoid (violet wireframe) guarantees "
        f"{scale.text(momentum.ellipsoid_inscribed, 3)} {units}. "
        "Sizing uses the blue radius.",
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

    scale = unit_scale(
        "N.m.s",
        [
            momentum.inscribed,
            wheels.usable_momentum_nms,
            *(analysis.assumptions.margin * d.required_nms for d in drivers),
        ],
    )
    units = unit_html(scale.units)
    fig, ax2 = plt.subplots(figsize=(9.0, 5.4))
    positions = np.arange(len(drivers))
    required = [
        scale.value(analysis.assumptions.margin * d.required_nms) for d in drivers
    ]
    usable = scale.value(wheels.usable_momentum_nms)
    verdicts = [usable >= r for r in required]
    ax2.bar(
        positions,
        required,
        color=[verdict_color(v) for v in verdicts],
        label=f"Required \u00d7 {analysis.assumptions.margin:g} margin",
    )
    threshold_line(
        ax2,
        usable,
        f"Usable envelope {scale.text(wheels.usable_momentum_nms, 3)} {units}",
        color=STRUCTURE_COLOR,
        ls="-",
        lw=LINE_WIDTH,
    )
    threshold_line(
        ax2,
        scale.value(momentum.inscribed),
        f"Wheel hardware $r_{{in}}$ {scale.text(momentum.inscribed, 3)} {units}",
        color=SERIES_1,
        lw=LINE_WIDTH,
    )
    for x, (driver, value, ok) in enumerate(
        zip(drivers, required, verdicts, strict=True)
    ):
        annotate_measurement(
            ax2,
            x,
            value,
            f"{scale.text(driver.required_nms, 3)} {units}",
            ok,
            offset=(4, 6),
        )
    ax2.set_yscale("log")
    ax2.set_xticks(positions)
    # "D1b post-B-dot handover" onto two lines, the second capitalised: a label
    # is a label, not the middle of a sentence.
    ax2.set_xticklabels(
        [
            f"{d.name.split(' ', 1)[0]}\n{_upper_first(d.name.split(' ', 1)[1])}"
            if " " in d.name
            else d.name
            for d in drivers
        ],
        fontsize=8,
    )
    ax2.set_ylabel(f"Momentum [{units}], log scale")
    ax2.set_ylim(min(required) / 5.0, scale.value(momentum.inscribed) * 5.0)
    _grid(ax2, which="both")
    ax2.legend(fontsize=8, loc="lower left")
    verdict_title(
        ax2,
        f"Required Momentum vs What the Vehicle May Use, {analysis.vehicle.name}",
        passed,
        "A bar above the usable-envelope line is a driver the certified "
        "envelope cannot hold",
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
    # Chosen from the bars rather than fixed at micro: a quieter vehicle's
    # budget lands a decade down and would read as a row of zeros.
    scale = unit_scale("N.m", [t.torque_nm for t in terms] + [budget.total_nm])
    units = unit_html(scale.units)
    secular = np.array([scale.value(t.secular_nm) for t in terms])
    cyclic = np.array([scale.value(t.cyclic_nm) for t in terms])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # Series colours, not verdict ones: secular and cyclic are two categories of
    # the same measurement, and drawing a category in the FAIL red would read as
    # a verdict this bar does not carry.
    ax.bar(positions, secular, color=SERIES_1, label="Secular (sizes desaturation)")
    ax.bar(
        positions,
        cyclic,
        bottom=secular,
        color=SERIES_2,
        label="Cyclic (sizes storage)",
    )
    for x, term in enumerate(terms):
        ax.text(
            x,
            scale.value(term.torque_nm),
            scale.text(term.torque_nm, 3),
            ha="center",
            va="bottom",
            fontsize=8,
            family=MONO,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_upper_first(t.name).replace(" ", "\n") for t in terms], fontsize=8
    )
    ax.set_ylabel(f"Disturbance torque [{units}]")
    _grid(ax)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Disturbance-Torque Budget, {analysis.vehicle.name}\n"
        f"Total {scale.text(budget.total_nm, 3)} {units} = secular "
        f"{scale.text(budget.secular_nm, 3)} + cyclic "
        f"{scale.text(budget.cyclic_nm, 3)}",
        fontsize=9,
    )

    margin = analysis.assumptions.margin
    authority = unit_scale(
        "N.m", [margin * budget.secular_nm, analysis.mtq.average_torque_nm]
    )
    authority_units = unit_html(authority.units)
    required = authority.value(margin * budget.secular_nm)
    available = authority.value(analysis.mtq.average_torque_nm)
    ok = available >= required
    ax2.bar([0, 1], [required, available], color=[verdict_color(ok)] * 2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        [f"Secular \u00d7 {margin:g} margin", "Rod average authority"], fontsize=9
    )
    ax2.set_yscale("log")
    ax2.set_ylabel(f"Torque [{authority_units}], log scale")
    _grid(ax2, which="both")
    annotate_measurement(ax2, 1, available, f"{available:.3g} {authority_units}", ok)
    verdict_title(
        ax2,
        "M1 Desaturation Authority",
        ok,
        "The rods must beat the secular torque, or the wheels saturate\n"
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
    scale = unit_scale("N.m.s", [removable, required])
    units = unit_html(scale.units)
    ax.bar(
        [0, 1],
        [scale.value(required), scale.value(removable)],
        color=[verdict_color(ok)] * 2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Tip-off momentum \u00d7 {margin:g}", "Removable in the budget"], fontsize=9
    )
    ax.set_yscale("log")
    ax.set_ylabel(f"Momentum [{units}], log scale")
    _grid(ax, which="both")
    annotate_measurement(
        ax,
        1,
        scale.value(removable),
        f"{scale.text(removable, 3)} {units}",
        ok,
    )
    verdict_title(
        ax,
        f"M2 Detumble Authority, {vehicle.name}",
        ok,
        f"Implied fast-phase duration {mtq.implied_detumble_s:.0f} s of a "
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
        color=[verdict_color(exit_ok), SERIES_1, SERIES_1],
    )
    threshold_line(
        ax2,
        rates[2],
        f"Noise floor at $|B|_{{min}}$ ({rates[2]:.2f} deg/s)",
        # A threshold is structure, not a verdict: drawing it in the status red
        # would read as a failure the line does not assert.
        color=STRUCTURE_COLOR,
        lw=LINE_WIDTH,
    )
    for x, value in enumerate(rates):
        ax2.text(
            x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, family=MONO
        )
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(
        ["DetumbleExitRadps", "Floor at $|B|_{mean}$", "Floor at $|B|_{min}$"],
        fontsize=8,
    )
    ax2.set_ylabel("Body rate [deg/s]")
    _grid(ax2)
    ax2.legend(fontsize=8, loc="upper left")
    verdict_title(
        ax2,
        "M3 B-dot Measurement Floor",
        exit_ok,
        "An exit threshold under the floor declares detumble complete on noise",
    )
    return save(fig, _prepare(out_dir) / "magnetorquer_sizing.png")


def write_text_report(
    analysis: SizingAnalysis,
    out_dir: str | Path | None = None,
    config_path: str | Path = "",
) -> Path:
    """Write ``<out_dir>/sizing_report.txt``: criteria, budget and justifications.

    Split out of :func:`write_all` because the rendered report is **not** a
    figure: it is the record, and a run that skips the figures must still leave
    it behind. The CLI calls this directly under ``--no-plots``.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.
    out_dir : str or pathlib.Path, optional
        Destination directory; created if absent.
    config_path : str or pathlib.Path, optional
        The config the vehicle came from, recorded in the report's provenance.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    report = sizing_report(
        analysis.vehicle, config_path, analysis.assumptions, analysis
    )
    target = _prepare(out_dir) / "sizing_report.txt"
    target.write_text(
        report.format_text()
        + "\n\n"
        + format_budget(analysis)
        + "\n\n"
        + format_derived(analysis)
        + "\n"
    )
    return target


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
    return [*figures, write_text_report(analysis, directory, config_path)]
