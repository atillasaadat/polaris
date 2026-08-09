"""The sizing figures as interactive plotly objects.

The matplotlib figures in :mod:`analysis.sizing.plots` stay: they are the record
written beside the report, and they print. These are the same verdicts drawn for
a reader who can rotate them — which the momentum envelope, a three-dimensional
achievable set with a second surface nested inside it, genuinely needs.

Nothing here computes a verdict. Every PASS/FAIL word and every number is read
from the :class:`~analysis.sizing.report.SizingAnalysis` and the
:class:`~analysis.common.report.AnalysisReport` handed in, per
``analysis/CLAUDE.md``: rendered output is never the verdict. The standing
plotting convention holds unchanged — thresholds are on the axes, measured
values are annotated where they were measured, and **colour is never
load-bearing alone**: every verdict is also the word PASS or FAIL, in the bar
label and in the hover text.

Units at the presentation boundary
----------------------------------
SI internally; N·m·s, µN·m and deg/s on the figures.

References
----------
Design doc §12 (analysis tools), §21.2 (generated artifacts).
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

from analysis.common.report import AnalysisReport
from analysis.sizing.envelope import Envelope
from analysis.sizing.mathfmt import _num, sentence_case
from analysis.sizing.report import SizingAnalysis

#: Verdict colours. Never load-bearing alone — every verdict is also a word.
#: Reserved for a *verdict*: a bar that is a PASS or a FAIL, never a category
#: and never a reference line. Chosen against the page's **white** background
#: (see :mod:`analysis.sizing.html`), and checked to stay separable under
#: protanopia and deuteranopia, where a green/amber pairing is not.
PASS_COLOR = "#0d5226"
FAIL_COLOR = "#e5484d"

#: The categorical pair, assigned in a fixed order and never cycled: series 1 is
#: always the first category a figure introduces, series 2 the second. Both
#: figures that use them label their series directly as well, so the colour is
#: an aid to grouping rather than the key to reading the chart.
SERIES_1 = "#0969da"
SERIES_2 = "#bc4c00"

#: Structure, not data: the deep slate the page uses for its rules and table
#: heads, here for the hardware zonotope, which is the frame the data sits in.
STRUCTURE_COLOR = "#24364a"
ZONOTOPE_COLOR = STRUCTURE_COLOR
#: The L2 ellipsoid: a third surface, muted so it recedes behind the two the
#: figure is actually comparing.
ELLIPSOID_COLOR = "#6a3d9a"
#: The certified ceiling, the second capability surface the reader compares.
USABLE_COLOR = SERIES_2

#: Light-theme figure furniture, so every figure agrees with the page and with
#: each other regardless of the reader's OS colour-scheme setting.
PAPER_BG = "#fcfcfb"
PLOT_BG = "#ffffff"
GRID_COLOR = "#e4e2dd"
ZERO_COLOR = "#c9c6c0"
FONT_COLOR = "#1a1d21"
AXIS_COLOR = "#5b6470"

#: Largest actuator count the zonotope hull is drawn for; the vertex set is
#: :math:`2^N`. Mirrors ``plots.MAX_HULL_WHEELS``.
MAX_HULL_ACTUATORS = 12


# --------------------------------------------------------------------------
# Geometry helpers, shared by the two 3D envelope figures
# --------------------------------------------------------------------------


def _hull_edge_lines(vertices: np.ndarray, hull: ConvexHull) -> go.Scatter3d:
    """Every unique hull edge as one polyline trace.

    A translucent mesh with no wireframe reads as a cloud: the reader cannot see
    that the inscribed sphere and the L2 ellipsoid sit *inside* the polyhedron,
    which is the whole claim the figure is making. The edges give the hull a
    shape to be inside of.

    Each triangle contributes three edges and each edge belongs to two triangles,
    so the pairs are sorted and de-duplicated; the segments are then joined with
    ``None`` separators into a **single** trace rather than one trace per edge,
    which keeps the legend and the renderer sane on a 16-vertex hull.
    """
    edges = {
        tuple(sorted(pair))
        for simplex in hull.simplices
        for pair in (
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[2], simplex[0]),
        )
    }
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for a, b in sorted(edges):
        x += [vertices[a, 0], vertices[b, 0], None]
        y += [vertices[a, 1], vertices[b, 1], None]
        z += [vertices[a, 2], vertices[b, 2], None]
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": ZONOTOPE_COLOR, "width": 2},
        opacity=0.55,
        name="Zonotope edges",
        showlegend=True,
        hoverinfo="skip",
    )


def _zonotope_mesh(env: Envelope, name: str) -> list[go.Mesh3d | go.Scatter3d]:
    """The achievable set as a translucent hull **and its edges**, or nothing if too large."""
    if env.n_actuators > MAX_HULL_ACTUATORS:  # pragma: no cover - no such vehicle
        return []
    signs = np.array(list(product((-1.0, 1.0), repeat=env.n_actuators)))
    vertices = env.capacity * signs @ env.axes.T
    hull = ConvexHull(vertices)
    return [
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            color=ZONOTOPE_COLOR,
            # Faint enough that the two inner surfaces read through it; the
            # edges below are what make the outer shape legible, not the fill.
            opacity=0.10,
            flatshading=True,
            name=name,
            showlegend=True,
            hoverinfo="name",
        ),
        _hull_edge_lines(vertices, hull),
    ]


def _sphere(radius: float, color: str, name: str, opacity: float) -> go.Surface:
    """A radius as a surface, so "guaranteed in every direction" is a shape."""
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    return go.Surface(
        x=radius * np.outer(np.cos(u), np.sin(v)),
        y=radius * np.outer(np.sin(u), np.sin(v)),
        z=radius * np.outer(np.ones_like(u), np.cos(v)),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        name=name,
        showlegend=True,
        hoverinfo="name",
    )


def _ellipsoid(env: Envelope, opacity: float = 0.28) -> go.Surface:
    """The L2 (minimum-norm) allocator's reach, inscribed in the zonotope.

    The opacity is per-figure: the three nested surfaces have to stay tellable
    apart at a glance, and how transparent the middle one must be depends on how
    many surfaces sit outside it.
    """
    semi = env.ellipsoid_semi_axes
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    return go.Surface(
        x=semi[0] * np.outer(np.cos(u), np.sin(v)),
        y=semi[1] * np.outer(np.sin(u), np.sin(v)),
        z=semi[2] * np.outer(np.ones_like(u), np.cos(v)),
        colorscale=[[0.0, ELLIPSOID_COLOR], [1.0, ELLIPSOID_COLOR]],
        showscale=False,
        opacity=opacity,
        name="L2 ellipsoid, if AllocMethodSel were 0",
        showlegend=True,
        hoverinfo="name",
    )


def _demand_vector(
    direction: np.ndarray, magnitude: float, label: str, inside: bool, units: str
) -> go.Scatter3d:
    """One requirement as a labelled arrow from the origin, verdict in words.

    The arrow is drawn in ink whatever its verdict. What the figure asks the
    reader to see is *geometric* — whether the vector ends inside the surfaces
    or outside them — and colouring the vector by the answer would let a reader
    take the verdict from the legend without ever looking at the geometry the
    figure exists to show. The word is in the label and in the hover text.
    """
    tip = np.asarray(direction, dtype=float) * magnitude
    verdict = (
        "Inside the capability envelope"
        if inside
        else "OUTSIDE the capability envelope"
    )
    return go.Scatter3d(
        x=[0.0, tip[0]],
        y=[0.0, tip[1]],
        z=[0.0, tip[2]],
        mode="lines+markers+text",
        text=["", label],
        textposition="top center",
        textfont={"size": 10},
        # Thicker than any surface edge on the figure: the drivers are the data
        # and must read on top of the three translucent envelopes behind them.
        line={"color": FONT_COLOR, "width": 8},
        marker={"size": [1, 6], "color": FONT_COLOR},
        textfont_color=FONT_COLOR,
        name=f"{label}: {_num(magnitude)} {units}, {verdict}",
        hovertemplate=f"{label}<br>{_num(magnitude)} {units}<br>{verdict}<extra></extra>",
    )


def _scene(title: str, axis_label: str) -> dict:
    """Equal-aspect 3D scene settings shared by both envelope figures.

    Explicitly light: the page is light unconditionally, so the figures state
    their own background, grid and font colours rather than inheriting whatever
    the reader's browser would supply.
    """
    axis = {
        "backgroundcolor": PLOT_BG,
        "showbackground": True,
        "gridcolor": GRID_COLOR,
        "zerolinecolor": ZERO_COLOR,
        "color": AXIS_COLOR,
    }
    return {
        "title": {"text": title, "font": {"size": 13, "color": FONT_COLOR}},
        "font": {"color": FONT_COLOR},
        "scene": {
            "xaxis": {**axis, "title": axis_label.format("x")},
            "yaxis": {**axis, "title": axis_label.format("y")},
            "zaxis": {**axis, "title": axis_label.format("z")},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "height": 620,
        "legend": {"orientation": "h", "y": -0.02, "font": {"size": 10}},
        "paper_bgcolor": PAPER_BG,
        "plot_bgcolor": PAPER_BG,
    }


# --------------------------------------------------------------------------
# The four interactive figures
# --------------------------------------------------------------------------


def momentum_envelope_figure(analysis: SizingAnalysis) -> go.Figure:
    """The headline: the momentum envelope in 3D with every driver on it.

    Four surfaces and the drivers. The distinction the figure exists to carry is
    between the **hardware** zonotope — what the wheels can hold — and the
    **usable** sphere at ``MomentumEnvelopeNms`` — what the certified analysis
    covers. Momentum the vehicle raises an envelope event over is momentum it
    does not have, so the drivers are judged against the smaller one.

    Each driver is drawn along the array's *weakest* direction, which is where
    its inscribed radius is attained: comparing a required magnitude against a
    guarantee is a comparison in the binding direction, and drawing it anywhere
    else would flatter the design.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    wheels = analysis.wheels
    env = wheels.momentum
    margin = analysis.assumptions.margin
    traces: list[go.Scatter3d | go.Surface | go.Mesh3d] = list(
        _zonotope_mesh(
            env, f"Wheel zonotope, hardware, best {_num(env.circumscribed)} N·m·s"
        )
    )
    traces.append(
        _sphere(
            env.inscribed,
            SERIES_1,
            f"Hardware guarantee rᵢₙ = {_num(env.inscribed)} N·m·s",
            0.13,
        )
    )
    traces.append(
        _sphere(
            wheels.usable_momentum_nms,
            USABLE_COLOR,
            f"Usable envelope = {_num(wheels.usable_momentum_nms)} N·m·s"
            + (" (MomentumEnvelopeNms binds)" if wheels.envelope_limited else ""),
            0.6,
        )
    )
    traces.append(_ellipsoid(env))
    for driver in wheels.drivers:
        if not driver.judged:
            continue
        required = margin * driver.required_nms
        traces.append(
            _demand_vector(
                env.worst_direction,
                required,
                driver.name.split(" ")[0],
                wheels.usable_momentum_nms >= required,
                "N·m·s",
            )
        )
    return go.Figure(
        data=traces,
        layout=_scene(
            f"Wheel momentum envelope, {analysis.vehicle.name} "
            f"(drivers at ×{margin:g} margin, along the weakest direction)",
            "h<sub>{}</sub> [N·m·s]",
        ),
    )


def torque_envelope_figure(analysis: SizingAnalysis) -> go.Figure:
    """The wheel torque envelope in 3D with the torque demand on it.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    env = analysis.wheels.torque
    margin = analysis.assumptions.margin
    required = margin * analysis.wheels.required_torque_nm
    traces: list[go.Scatter3d | go.Surface | go.Mesh3d] = list(
        _zonotope_mesh(env, f"Torque zonotope, best {_num(env.circumscribed)} N·m")
    )
    traces.append(
        _sphere(
            env.inscribed,
            # A capability surface, so it takes a series colour and not a
            # verdict one; the demand arrow's label carries the verdict.
            SERIES_1,
            f"Guarantee rᵢₙ = {_num(env.inscribed)} N·m",
            0.22,
        )
    )
    traces.append(_ellipsoid(env, opacity=0.34))
    traces.append(
        _demand_vector(
            env.worst_direction,
            required,
            "Torque demand",
            env.inscribed >= required,
            "N·m",
        )
    )
    return go.Figure(
        data=traces,
        layout=_scene(
            f"Wheel torque envelope, {analysis.vehicle.name} "
            f"(demand = PidMaxTorqueNm + disturbance, ×{margin:g} margin)",
            "\u03c4<sub>{}</sub> [N·m]",
        ),
    )


def disturbance_figure(analysis: SizingAnalysis) -> go.Figure:
    """The §5.3 budget per source, secular against cyclic, with rod authority.

    Parameters
    ----------
    analysis : SizingAnalysis
        The computed analysis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    budget = analysis.budget
    # Capitalised here rather than in the budget: these are axis labels on a
    # figure, and the term names are written for a console table.
    names = [t.name[:1].upper() + t.name[1:] for t in budget.terms]
    available = analysis.mtq.average_torque_nm * 1e6
    secular = [t.secular_nm * 1e6 for t in budget.terms]
    cyclic = [t.cyclic_nm * 1e6 for t in budget.terms]
    # The log axis is set explicitly. Left to autorange it has to accommodate a
    # text label on every bar as well as the authority line far above them, and
    # plotly resolves that on a log scale by opening up dozens of empty decades,
    # which flattens every bar to the axis.
    drawn = [v for v in [*secular, *cyclic] if v > 0.0] + [available]
    span = [math.log10(min(drawn) / 6.0), math.log10(max(drawn) * 4.0)]
    # Direct labels, so the two series are told apart by reading rather than by
    # matching a colour back to a legend swatch. A term with no contribution of
    # that kind gets no label instead of a "0" cluttering the axis.
    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=secular,
                name="Secular, sizes desaturation",
                marker_color=SERIES_1,
                text=[f"Secular {v:.3g}" if v > 0.0 else "" for v in secular],
                textposition="outside",
                textangle=0,
                textfont={"size": 10, "color": FONT_COLOR},
                customdata=[t.formula for t in budget.terms],
                hovertemplate="<b>%{x}</b><br>Secular %{y:.4g} µN·m<br>%{customdata}"
                "<extra></extra>",
            ),
            go.Bar(
                x=names,
                y=cyclic,
                name="Cyclic, sizes storage",
                marker_color=SERIES_2,
                text=[f"Cyclic {v:.3g}" if v > 0.0 else "" for v in cyclic],
                textposition="outside",
                textangle=0,
                textfont={"size": 10, "color": FONT_COLOR},
                customdata=[t.formula for t in budget.terms],
                hovertemplate="<b>%{x}</b><br>Cyclic %{y:.4g} µN·m<br>%{customdata}"
                "<extra></extra>",
            ),
        ]
    )
    # Ink, not a verdict colour: the rods' authority is the reference this chart
    # is read against, and it is no more a PASS than an axis is.
    fig.add_hline(
        y=available,
        line={"color": FONT_COLOR, "width": 2, "dash": "dash"},
        annotation_text=f"Magnetorquer desaturation authority {available:.3g} µN·m "
        "(orbit-average, worst direction, weakest field)",
        # Below the line, not above it: the line sits near the top of the range
        # and an annotation above it lands outside the plotting area.
        annotation_position="bottom left",
        annotation_font_size=11,
        annotation_font_color=FONT_COLOR,
        annotation_bgcolor=PAPER_BG,
    )
    fig.update_layout(
        barmode="group",
        # Thin bars with a visible gap of surface between adjacent fills, so a
        # pair reads as two marks rather than one two-tone block.
        bargap=0.42,
        bargroupgap=0.08,
        title={
            "text": f"Disturbance-torque budget: total {budget.total_nm * 1e6:.3g} "
            f"µN·m = secular {budget.secular_nm * 1e6:.3g} + cyclic "
            f"{budget.cyclic_nm * 1e6:.3g} µN·m.<br>The rods must beat the "
            "<b>secular total</b>, or the wheels saturate whatever their size.",
            "font": {"size": 13, "color": FONT_COLOR},
        },
        yaxis_title="Disturbance torque [µN·m], log scale",
        height=460,
        margin={"l": 60, "r": 20, "t": 70, "b": 40},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": FONT_COLOR},
        xaxis={"color": AXIS_COLOR, "gridcolor": GRID_COLOR},
        # Log scale, and not for aesthetics: the rods' authority is ~600x the
        # disturbance they must beat on this vehicle, so a linear axis scaled to
        # show the threshold line renders every actual bar as a flat zero — the
        # chart would say "there is no disturbance" when what it means is "the
        # margin is enormous". The ratio between the terms is the readable
        # quantity here, which is what a log axis shows.
        yaxis={
            "type": "log",
            "range": span,
            # Decades only: the 2 and 5 minor labels a log axis defaults to are
            # noise on a chart whose point is the ratio between the bars.
            "dtick": 1,
            "color": AXIS_COLOR,
            "gridcolor": GRID_COLOR,
            "zerolinecolor": ZERO_COLOR,
        },
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def margin_figure(report: AnalysisReport, order: list | None = None) -> go.Figure:
    """Every criterion's margin as a horizontal bar, zero marked, verdict in words.

    Parameters
    ----------
    report : analysis.common.report.AnalysisReport
        The structured report; the verdicts drawn are its own.
    order : list of analysis.common.report.Criterion, optional
        The criteria in the order the reader should meet them, top to bottom.
        The page passes its own document order — grouped by family, worst margin
        first inside each family — so the chart and the criteria table above it
        tell the same story in the same sequence. Defaults to the report's own
        order. Must be criteria of ``report``; nothing here is recomputed.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Plotly stacks a horizontal bar chart bottom-up, so the sequence is
    # reversed to put the first criterion at the top.
    criteria = list(order if order is not None else report.criteria)[::-1]
    # A margin of +4000 % and one of +40 % are both passes and a linear axis
    # renders the second as nothing, so the bars are clipped for *drawing* only
    # — the label beside each bar always carries the true number.
    clipped = [max(-200.0, min(400.0, c.margin_pct)) for c in criteria]
    labels = [
        f"{'PASS' if c.passes else 'FAIL'} {_num(c.margin_pct, 3)} %" for c in criteria
    ]
    fig = go.Figure(
        go.Bar(
            x=clipped,
            y=[sentence_case(c.name) for c in criteria],
            orientation="h",
            marker_color=[PASS_COLOR if c.passes else FAIL_COLOR for c in criteria],
            text=labels,
            textposition="outside",
            textfont={"size": 11, "color": FONT_COLOR},
            customdata=[
                [
                    "PASS" if c.passes else "FAIL",
                    f"{'≥' if c.sense == 'min' else '≤'} {_num(c.threshold)} {c.units}",
                    f"{_num(c.measured)} {c.units}",
                ]
                for c in criteria
            ],
            hovertemplate="%{y}<br><b>%{customdata[0]}</b><br>"
            "Threshold %{customdata[1]}<br>Measured %{customdata[2]}<extra></extra>",
        )
    )
    # Zero is the threshold every bar is measured against, so it is drawn solid
    # in ink rather than as another piece of grid.
    fig.add_vline(x=0.0, line={"color": FONT_COLOR, "width": 2})
    fig.update_layout(
        bargap=0.42,
        title={
            "text": "Margin per criterion, worst first within each family "
            "(bars clipped to ±400 % for legibility; the label carries the "
            "true value)",
            "font": {"size": 13, "color": FONT_COLOR},
        },
        xaxis_title="Margin [% of threshold]",
        height=max(360, 34 * len(criteria) + 140),
        margin={"l": 340, "r": 90, "t": 60, "b": 50},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": FONT_COLOR},
        xaxis={
            "color": AXIS_COLOR,
            "gridcolor": GRID_COLOR,
            "zerolinecolor": ZERO_COLOR,
        },
        yaxis={"color": AXIS_COLOR, "gridcolor": GRID_COLOR},
        showlegend=False,
    )
    return fig
