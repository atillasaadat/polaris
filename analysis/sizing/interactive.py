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
from analysis.sizing.report import SizingAnalysis

#: Verdict colours, matched to :mod:`analysis.common.plotting` so the HTML and
#: the PNGs agree. Never load-bearing alone — every verdict is also a word.
#: Chosen against the page's **white** background (see :mod:`analysis.sizing.html`):
#: each is dark enough to hold contrast when drawn translucent over white, which
#: a palette picked against a dark canvas is not.
PASS_COLOR = "#1a7f37"
FAIL_COLOR = "#b3261e"
ZONOTOPE_COLOR = "#1f4e79"
ELLIPSOID_COLOR = "#6a3d9a"
USABLE_COLOR = "#b26800"

#: Light-theme figure furniture, so every figure agrees with the page and with
#: each other regardless of the reader's OS colour-scheme setting.
PAPER_BG = "#ffffff"
PLOT_BG = "#fbfcfd"
GRID_COLOR = "#dfe3e8"
ZERO_COLOR = "#b9c1ca"
FONT_COLOR = "#16191d"
AXIS_COLOR = "#454b53"

#: Largest actuator count the zonotope hull is drawn for; the vertex set is
#: :math:`2^N`. Mirrors ``plots.MAX_HULL_WHEELS``.
MAX_HULL_ACTUATORS = 12


def _num(value: float, digits: int = 4) -> str:
    """A number a reader can scan, across nine orders of magnitude."""
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "∞" if value > 0 else "−∞"
    if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e5):
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


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
        name="zonotope edges",
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
        name="L2 ellipsoid (if AllocMethodSel were 0)",
        showlegend=True,
        hoverinfo="name",
    )


def _demand_vector(
    direction: np.ndarray, magnitude: float, label: str, inside: bool, units: str
) -> go.Scatter3d:
    """One requirement as a labelled arrow from the origin, verdict in words."""
    tip = np.asarray(direction, dtype=float) * magnitude
    verdict = "inside the usable envelope" if inside else "OUTSIDE the usable envelope"
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
        line={"color": PASS_COLOR if inside else FAIL_COLOR, "width": 8},
        marker={"size": [1, 6], "color": PASS_COLOR if inside else FAIL_COLOR},
        textfont_color=FONT_COLOR,
        name=f"{label} — {_num(magnitude)} {units}, {verdict}",
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
            env, f"wheel zonotope (hardware, best {_num(env.circumscribed)})"
        )
    )
    traces.append(
        _sphere(
            env.inscribed,
            ZONOTOPE_COLOR,
            f"hardware guarantee r_in = {_num(env.inscribed)} N.m.s",
            0.13,
        )
    )
    traces.append(
        _sphere(
            wheels.usable_momentum_nms,
            USABLE_COLOR,
            f"USABLE envelope = {_num(wheels.usable_momentum_nms)} N.m.s"
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
                "N.m.s",
            )
        )
    return go.Figure(
        data=traces,
        layout=_scene(
            f"{analysis.vehicle.name} — wheel momentum envelope "
            f"(drivers shown at ×{margin:g} margin, along the weakest direction)",
            "h_{} [N.m.s]",
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
        _zonotope_mesh(env, f"torque zonotope (best {_num(env.circumscribed)} N.m)")
    )
    traces.append(
        _sphere(
            env.inscribed,
            PASS_COLOR if env.inscribed >= required else FAIL_COLOR,
            f"guarantee r_in = {_num(env.inscribed)} N.m",
            0.22,
        )
    )
    traces.append(_ellipsoid(env, opacity=0.34))
    traces.append(
        _demand_vector(
            env.worst_direction,
            required,
            "torque demand",
            env.inscribed >= required,
            "N.m",
        )
    )
    return go.Figure(
        data=traces,
        layout=_scene(
            f"{analysis.vehicle.name} — wheel torque envelope "
            f"(demand = PidMaxTorqueNm + disturbance, ×{margin:g} margin)",
            "tau_{} [N.m]",
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
    names = [t.name for t in budget.terms]
    available = analysis.mtq.average_torque_nm * 1e6
    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=[t.secular_nm * 1e6 for t in budget.terms],
                name="secular (sizes desaturation)",
                marker_color=FAIL_COLOR,
                customdata=[t.formula for t in budget.terms],
                hovertemplate="%{x} secular<br>%{y:.4g} uN.m<br>%{customdata}<extra></extra>",
            ),
            go.Bar(
                x=names,
                y=[t.cyclic_nm * 1e6 for t in budget.terms],
                name="cyclic (sizes storage)",
                marker_color=ZONOTOPE_COLOR,
                customdata=[t.formula for t in budget.terms],
                hovertemplate="%{x} cyclic<br>%{y:.4g} uN.m<br>%{customdata}<extra></extra>",
            ),
        ]
    )
    fig.add_hline(
        y=available,
        line={"color": PASS_COLOR, "width": 2, "dash": "dash"},
        annotation_text=f"MTQ desaturation authority {available:.3g} uN.m "
        "(orbit-average, worst direction, weakest field)",
        annotation_position="top left",
        annotation_font_size=11,
    )
    fig.update_layout(
        barmode="group",
        title={
            "text": f"Disturbance-torque budget — total {budget.total_nm * 1e6:.3g} "
            f"uN.m = secular {budget.secular_nm * 1e6:.3g} + cyclic "
            f"{budget.cyclic_nm * 1e6:.3g}. The rods must beat the <b>secular "
            "total</b>, or the wheels saturate whatever their size.",
            "font": {"size": 13, "color": FONT_COLOR},
        },
        yaxis_title="disturbance torque [uN.m] — log scale",
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
            "color": AXIS_COLOR,
            "gridcolor": GRID_COLOR,
            "zerolinecolor": ZERO_COLOR,
        },
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def margin_figure(report: AnalysisReport) -> go.Figure:
    """Every criterion's margin as a horizontal bar, zero marked, verdict in words.

    Parameters
    ----------
    report : analysis.common.report.AnalysisReport
        The structured report; the verdicts drawn are its own.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    criteria = list(report.criteria)[::-1]
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
            y=[c.name for c in criteria],
            orientation="h",
            marker_color=[PASS_COLOR if c.passes else FAIL_COLOR for c in criteria],
            text=labels,
            textposition="outside",
            customdata=[
                [
                    "PASS" if c.passes else "FAIL",
                    f"{'≥' if c.sense == 'min' else '≤'} {_num(c.threshold)} {c.units}",
                    f"{_num(c.measured)} {c.units}",
                ]
                for c in criteria
            ],
            hovertemplate="%{y}<br><b>%{customdata[0]}</b><br>"
            "threshold %{customdata[1]}<br>measured %{customdata[2]}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line={"color": AXIS_COLOR, "width": 2})
    fig.update_layout(
        title={
            "text": "Margin per criterion — bars clipped to ±400 % for legibility; "
            "the label carries the true value",
            "font": {"size": 13, "color": FONT_COLOR},
        },
        xaxis_title="margin [% of threshold]",
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
