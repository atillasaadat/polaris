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
SI internally; the figures pick an SI prefix per axis from the magnitudes they
carry (:func:`analysis.sizing.mathfmt.unit_scale`), so a wheel envelope is drawn
in mN·m·s rather than in four leading zeros, and rates are in deg/s. One prefix
per figure: the surfaces and the arrows on an envelope figure are compared by
eye and must therefore be compared in one unit.

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

from analysis.common.figures import (
    AXIS_COLOR,
    FAIL_COLOR,
    FONT_COLOR,
    GRID_COLOR,
    PAPER_BG,
    PASS_COLOR,
    PLOT_BG,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    STRUCTURE_COLOR,
    ZERO_COLOR,
)
from analysis.common.report import AnalysisReport
from analysis.sizing.envelope import Envelope
from analysis.sizing.mathfmt import (
    _num,
    criterion_label,
    sentence_case,
    unit_html,
    unit_scale,
)
from analysis.sizing.report import SizingAnalysis

# The palette and the light-theme furniture are the page's, not this module's:
# two figures on one page must agree about what white is, and about which colour
# means a verdict. `analysis.common.figures` states them once.
#
# Two names stay local because they are sizing's own vocabulary rather than the
# shared palette: which *surface* gets which series is a fact about the momentum
# envelope figure and means nothing to another report.
ZONOTOPE_COLOR = STRUCTURE_COLOR
#: The L2 ellipsoid: a third surface, muted so it recedes behind the two the
#: figure is actually comparing.
ELLIPSOID_COLOR = SERIES_3
#: The certified ceiling, the second capability surface the reader compares.
USABLE_COLOR = SERIES_2

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


def _zonotope_mesh(
    env: Envelope, name: str, factor: float = 1.0
) -> list[go.Mesh3d | go.Scatter3d]:
    """The achievable set as a translucent hull **and its edges**, or nothing if too large.

    @p factor scales the geometry into the axis units the figure declares; it is
    a change of prefix and nothing else, so every shape on the figure takes the
    same one.
    """
    if env.n_actuators > MAX_HULL_ACTUATORS:  # pragma: no cover - no such vehicle
        return []
    signs = np.array(list(product((-1.0, 1.0), repeat=env.n_actuators)))
    vertices = factor * env.capacity * signs @ env.axes.T
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


def _ellipsoid(env: Envelope, opacity: float = 0.28, factor: float = 1.0) -> go.Surface:
    """The L2 (minimum-norm) allocator's reach, inscribed in the zonotope.

    The opacity is per-figure: the three nested surfaces have to stay tellable
    apart at a glance, and how transparent the middle one must be depends on how
    many surfaces sit outside it.
    """
    semi = factor * env.ellipsoid_semi_axes
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
    direction: np.ndarray,
    magnitude: float,
    label: str,
    inside: bool,
    units: str,
    tip_label: str = "",
) -> go.Scatter3d:
    """One requirement as a labelled arrow from the origin, verdict in words.

    @p tip_label is what is written at the arrow's point, where three overlapping
    vectors leave room for a short code and nothing more; @p label is the full
    name, which the legend and the hover text carry.

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
        text=["", tip_label or label],
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
    # One prefix for the whole figure: the surfaces and the driver arrows are
    # compared by eye, so they must be compared in one unit.
    scale = unit_scale(
        "N.m.s",
        [
            env.circumscribed,
            env.inscribed,
            env.ellipsoid_inscribed,
            wheels.usable_momentum_nms,
            *(margin * d.required_nms for d in wheels.drivers if d.judged),
        ],
    )
    units = unit_html(scale.units)
    traces: list[go.Scatter3d | go.Surface | go.Mesh3d] = list(
        _zonotope_mesh(
            env,
            f"Wheel zonotope, hardware, best {scale.text(env.circumscribed)} {units}",
            scale.factor,
        )
    )
    traces.append(
        _sphere(
            scale.value(env.inscribed),
            SERIES_1,
            f"Hardware guarantee rᵢₙ = {scale.text(env.inscribed)} {units}",
            0.13,
        )
    )
    traces.append(
        _sphere(
            scale.value(wheels.usable_momentum_nms),
            USABLE_COLOR,
            f"Usable envelope = {scale.text(wheels.usable_momentum_nms)} {units}"
            + (" (MomentumEnvelopeNms binds)" if wheels.envelope_limited else ""),
            0.6,
        )
    )
    traces.append(_ellipsoid(env, factor=scale.factor))
    for driver in wheels.drivers:
        if not driver.judged:
            continue
        required = margin * driver.required_nms
        traces.append(
            _demand_vector(
                env.worst_direction,
                scale.value(required),
                criterion_label(driver.name),
                wheels.usable_momentum_nms >= required,
                units,
                tip_label=driver.name.split(" ")[0],
            )
        )
    return go.Figure(
        data=traces,
        layout=_scene(
            f"Wheel momentum envelope, {analysis.vehicle.name} "
            f"(drivers at ×{margin:g} margin, along the weakest direction)",
            f"h<sub>{{}}</sub> [{units}]",
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
    scale = unit_scale(
        "N.m", [env.circumscribed, env.inscribed, env.ellipsoid_inscribed, required]
    )
    units = unit_html(scale.units)
    traces: list[go.Scatter3d | go.Surface | go.Mesh3d] = list(
        _zonotope_mesh(
            env,
            f"Torque zonotope, best {scale.text(env.circumscribed)} {units}",
            scale.factor,
        )
    )
    traces.append(
        _sphere(
            scale.value(env.inscribed),
            # A capability surface, so it takes a series colour and not a
            # verdict one; the demand arrow's label carries the verdict.
            SERIES_1,
            f"Guarantee rᵢₙ = {scale.text(env.inscribed)} {units}",
            0.22,
        )
    )
    traces.append(_ellipsoid(env, opacity=0.34, factor=scale.factor))
    traces.append(
        _demand_vector(
            env.worst_direction,
            scale.value(required),
            "Torque demand",
            env.inscribed >= required,
            units,
        )
    )
    return go.Figure(
        data=traces,
        layout=_scene(
            f"Wheel torque envelope, {analysis.vehicle.name} "
            f"(demand = PidMaxTorqueNm + disturbance, ×{margin:g} margin)",
            f"\u03c4<sub>{{}}</sub> [{units}]",
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
    # One prefix for the axis, chosen from the bars themselves rather than fixed
    # at micro: a quieter vehicle's budget lands a decade down and would read as
    # a column of zeros.
    scale = unit_scale(
        "N.m",
        [t.torque_nm for t in budget.terms]
        + [budget.total_nm, analysis.mtq.average_torque_nm],
    )
    units = unit_html(scale.units)
    available = scale.value(analysis.mtq.average_torque_nm)
    secular = [scale.value(t.secular_nm) for t in budget.terms]
    cyclic = [scale.value(t.cyclic_nm) for t in budget.terms]
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
                hovertemplate=f"<b>%{{x}}</b><br>Secular %{{y:.4g}} {units}"
                "<br>%{customdata}<extra></extra>",
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
                hovertemplate=f"<b>%{{x}}</b><br>Cyclic %{{y:.4g}} {units}"
                "<br>%{customdata}<extra></extra>",
            ),
        ]
    )
    # Ink, not a verdict colour: the rods' authority is the reference this chart
    # is read against, and it is no more a PASS than an axis is.
    fig.add_hline(
        y=available,
        line={"color": FONT_COLOR, "width": 2, "dash": "dash"},
        annotation_text=f"Magnetorquer desaturation authority {available:.3g} {units} "
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
            "text": f"Disturbance-torque budget: total {scale.text(budget.total_nm, 3)} "
            f"{units} = secular {scale.text(budget.secular_nm, 3)} + cyclic "
            f"{scale.text(budget.cyclic_nm, 3)} {units}.<br>The rods must beat the "
            "<b>secular total</b>, or the wheels saturate whatever their size.",
            "font": {"size": 13, "color": FONT_COLOR},
        },
        yaxis_title=f"Disturbance torque [{units}], log scale",
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
    hover_scales = [
        unit_scale(c.units, (c.threshold, c.measured, c.margin)) for c in criteria
    ]
    labels = [
        f"{'PASS' if c.passes else 'FAIL'} {_num(c.margin_pct, 3)} %" for c in criteria
    ]
    fig = go.Figure(
        go.Bar(
            x=clipped,
            y=[sentence_case(criterion_label(c.name)) for c in criteria],
            orientation="h",
            marker_color=[PASS_COLOR if c.passes else FAIL_COLOR for c in criteria],
            text=labels,
            textposition="outside",
            textfont={"size": 11, "color": FONT_COLOR},
            # One scale per criterion, so a row's threshold and measured value
            # are always quoted in the same unit and the comparison in the
            # hover is direct.
            customdata=[
                [
                    "PASS" if c.passes else "FAIL",
                    f"{'≥' if c.sense == 'min' else '≤'} {scale.text(c.threshold)}"
                    f" {unit_html(scale.units)}",
                    f"{scale.text(c.measured)} {unit_html(scale.units)}",
                ]
                for c, scale in zip(criteria, hover_scales, strict=True)
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
