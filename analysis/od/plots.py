"""The campaign's figures, as interactive plotly objects.

Four figures, each answering one of the campaign's questions and none of them a
bare curve the reader has to interpret. The standing convention holds: the
threshold is on the axes, the measured value is annotated where it was measured,
and **colour is never load-bearing alone** — every verdict is also a word.

Nothing here computes a verdict. Every band, every PASS and every FAIL is read
from :mod:`analysis.od.statistics` and the report handed in.

Why error is drawn against its own covariance and not alone
------------------------------------------------------------
An error trace on its own says how wrong the filter was. It cannot say whether
the filter *knew*, and that second question is the one that decides whether the
estimate is safe to fly on: an accurate but overconfident filter is the one that
will eventually reject a correct measurement. So the error is drawn inside the
±3σ envelope the filter claimed at the same instant, and the reader's eye checks
containment rather than magnitude. Where the two are drawn on a log axis it is
because a week of LEO OD spans metres during a fix stream and kilometres during
a six-hour outage, and a linear axis would show one of those and hide the other.

Downsampling, and why the worst case survives it
-------------------------------------------------
A 7-day run at 10 s is 60480 points per trace and a campaign is thirty of them;
handed to a browser whole, the page stops being openable. The traces are
therefore decimated — but by **envelope**, not by stride: each output bucket
keeps its own minimum and maximum, so the worst error in the arc is still drawn
at the sample it happened on. Plain striding would silently delete exactly the
spikes the figure exists to show.

References
----------
Design doc §8.3, §9.2, §13, §21.2 (generated artifacts).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from analysis.common.figures import (
    FAIL_COLOR,
    MUTED_COLOR,
    PASS_COLOR,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    STRUCTURE_COLOR,
    layout,
)
from analysis.od.records import ScenarioRun
from analysis.od.statistics import CampaignStatistics, ScenarioStatistics

__all__ = [
    "consistency_figure",
    "error_history_figure",
    "refusal_figure",
    "regime_figure",
]

#: Points per trace after decimation. Two per bucket (min and max), so the
#: bucket count is half this. Sized so a page with several traces stays
#: interactive on a laptop rather than to any property of the data.
MAX_POINTS = 2000


def _envelope_decimate(
    t_s: np.ndarray, values: np.ndarray, buckets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate ``values`` keeping each bucket's extremes, at their own times.

    Returns the points in time order, so the result is still a line rather than
    a zig-zag between unrelated instants.
    """
    count = t_s.size
    if count <= 2 * buckets or buckets < 1:
        return t_s, values
    edges = np.linspace(0, count, buckets + 1, dtype=int)
    keep: list[int] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        window = values[start:stop]
        finite = np.flatnonzero(np.isfinite(window))
        if finite.size == 0:
            keep.append(start)
            continue
        keep.append(start + int(finite[np.argmin(window[finite])]))
        keep.append(start + int(finite[np.argmax(window[finite])]))
    index = np.unique(np.array(keep, dtype=int))
    return t_s[index], values[index]


def error_history_figure(run: ScenarioRun) -> go.Figure:
    """One run's position error against the covariance the filter claimed.

    The headline figure. Three traces: the error, the ±3σ the filter published
    at the same instant, and a shaded band wherever the filter had no valid
    solution — which is a different state from a large error and must not read
    as one.

    Parameters
    ----------
    run : analysis.od.records.ScenarioRun

    Returns
    -------
    plotly.graph_objects.Figure
    """
    hours = run.t_s / 3600.0
    buckets = MAX_POINTS // 2

    # Only where a solution existed. A cycle with no solution has no error, and
    # plotting a zero there would draw the coast as the most accurate stretch of
    # the arc — the exact inversion the outage scenarios exist to catch.
    live = run.solution_valid
    err = np.where(live, run.pos_err_m, np.nan)
    sigma = np.where(live, 3.0 * run.pos_sigma_m, np.nan)

    t_err, y_err = _envelope_decimate(hours, err, buckets)
    t_sig, y_sig = _envelope_decimate(hours, sigma, buckets)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=t_sig,
            y=y_sig,
            name="Filter's own 3σ",
            mode="lines",
            line={"color": SERIES_2, "width": 1.4, "dash": "dot"},
            hovertemplate="%{x:.3f} h<br>3σ = %{y:.3g} m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=t_err,
            y=y_err,
            name="Position error vs truth",
            mode="lines",
            line={"color": SERIES_1, "width": 1.2},
            hovertemplate="%{x:.3f} h<br>|dr| = %{y:.3g} m<extra></extra>",
        )
    )

    for start, stop in _spans(~live, hours):
        figure.add_vrect(
            x0=start,
            x1=stop,
            fillcolor=FAIL_COLOR,
            opacity=0.10,
            line_width=0,
            layer="below",
        )

    worst = float(np.nanmax(err)) if np.isfinite(err).any() else float("nan")
    if np.isfinite(worst):
        at = float(hours[int(np.nanargmax(err))])
        figure.add_annotation(
            x=at,
            y=worst,
            text=f"worst {worst:.3g} m",
            showarrow=True,
            arrowhead=2,
            arrowcolor=STRUCTURE_COLOR,
            font={"size": 10, "color": STRUCTURE_COLOR},
        )

    figure.update_layout(
        **layout(
            f"{run.scenario} — run {run.run}: error inside its own covariance",
            "Time from epoch [h]",
            "Position [m]",
            height=380,
            log_y=True,
        )
    )
    return figure


def _spans(flag: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous ``True`` runs of ``flag`` as (start, stop) pairs of ``x``."""
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(flag):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((float(x[start]), float(x[index])))
            start = None
    if start is not None:
        spans.append((float(x[start]), float(x[-1])))
    return spans


def consistency_figure(stats: CampaignStatistics) -> go.Figure:
    """Every scenario's NEES and NIS against their chi-square acceptance bands.

    The bands are the figure's thresholds and are drawn as shaded regions rather
    than as lines, because what matters is being *inside* an interval, not on the
    right side of a bound. A marker outside its band carries the word FAIL in its
    label and its hover text; being above the band is called out as *optimistic*,
    since that is the direction that makes a filter unsafe rather than merely
    wasteful.

    Parameters
    ----------
    stats : analysis.od.statistics.CampaignStatistics

    Returns
    -------
    plotly.graph_objects.Figure
    """
    names = [entry.scenario for entry in stats.scenarios]
    figure = go.Figure()

    for offset, (key, colour, label) in enumerate(
        ((("nees"), SERIES_1, "NEES (6-state)"), (("nis"), SERIES_3, "NIS (position)"))
    ):
        intervals = [getattr(entry, key) for entry in stats.scenarios]
        # Normalised by the interval's own centre so NEES (about 6) and NIS
        # (about 3) share one axis and one acceptance band. The reader compares
        # each marker against 1.0, not against a different number per series.
        centre = [i.dof if np.isfinite(i.mean) else np.nan for i in intervals]
        figure.add_trace(
            go.Scatter(
                x=names,
                y=[i.mean / c if c else np.nan for i, c in zip(intervals, centre)],
                name=label,
                mode="markers",
                marker={
                    "size": 11,
                    "color": [
                        PASS_COLOR if i.consistent else FAIL_COLOR for i in intervals
                    ],
                    "symbol": "circle" if offset == 0 else "diamond",
                    "line": {"width": 1.5, "color": colour},
                },
                text=[
                    (
                        f"{label}: {i.mean:.3g} in [{i.lower:.3g}, {i.upper:.3g}] "
                        f"over {i.samples} runs — "
                        + (
                            "PASS"
                            if i.consistent
                            else (
                                "FAIL, optimistic"
                                if i.optimistic
                                else "FAIL, pessimistic"
                            )
                        )
                    )
                    for i in intervals
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        for index, interval in enumerate(intervals):
            if not np.isfinite(interval.mean):
                continue
            figure.add_shape(
                type="rect",
                x0=index - 0.32,
                x1=index + 0.32,
                y0=interval.lower / interval.dof,
                y1=interval.upper / interval.dof,
                fillcolor=MUTED_COLOR,
                opacity=0.18,
                line_width=0,
                layer="below",
            )

    figure.add_hline(
        y=1.0,
        line={"color": STRUCTURE_COLOR, "width": 1, "dash": "dash"},
        annotation_text="consistent",
        annotation_font_size=10,
    )
    figure.update_layout(
        **layout(
            "Filter consistency: measured average over its acceptance interval",
            "",
            "Average normalised error / degrees of freedom",
            height=380,
        )
    )
    return figure


def regime_figure(entry: ScenarioStatistics) -> go.Figure:
    """One scenario's error distribution, split by which fault was armed.

    Pooling the regimes would report neither: a scenario's nominal stretches and
    its outage stretches are different populations, and the average of the two is
    a number describing no state the vehicle is ever in. Quantiles rather than a
    mean and a standard deviation, because the error is bounded below by zero and
    has a long right tail wherever a fault is armed.

    Parameters
    ----------
    entry : analysis.od.statistics.ScenarioStatistics

    Returns
    -------
    plotly.graph_objects.Figure
    """
    regimes = list(entry.regimes)
    summaries = [entry.regimes[name].position for name in regimes]
    figure = go.Figure()
    for label, values, colour in (
        ("median", [s.median for s in summaries], SERIES_1),
        ("95th percentile", [s.p95 for s in summaries], SERIES_2),
        ("worst", [s.worst for s in summaries], SERIES_3),
    ):
        figure.add_trace(
            go.Bar(
                x=regimes,
                y=values,
                name=label,
                marker_color=colour,
                text=[f"{v:.3g} m" if np.isfinite(v) else "n/a" for v in values],
                textposition="outside",
                textfont={"size": 9},
                hovertemplate="%{x}<br>" + label + " = %{y:.4g} m<extra></extra>",
            )
        )
    figure.update_layout(
        **layout(
            f"{entry.scenario} — position error by regime",
            "",
            "Position error [m]",
            height=360,
            log_y=True,
        ),
        barmode="group",
    )
    return figure


def refusal_figure(stats: CampaignStatistics) -> go.Figure:
    """Which layer refused what, per scenario.

    The figure that exists because of a defect: a GEO-radius fix rejected by the
    innovation gate looks, in any count of accepted-versus-rejected, exactly like
    one refused at the trust boundary — and the two are not the same, because the
    innovation gate does not exist on the seed path. Stacking by refusal *name*
    is what makes the difference visible without reading the records.

    Parameters
    ----------
    stats : analysis.od.statistics.CampaignStatistics

    Returns
    -------
    plotly.graph_objects.Figure
    """
    names = [entry.scenario for entry in stats.scenarios]
    kinds = sorted(
        {
            kind
            for entry in stats.scenarios
            for summary in entry.regimes.values()
            for kind in summary.refusals
        }
    )
    palette = (SERIES_1, SERIES_2, SERIES_3, STRUCTURE_COLOR, MUTED_COLOR, FAIL_COLOR)

    figure = go.Figure()
    for index, kind in enumerate(kinds):
        counts = [
            sum(summary.refusals.get(kind, 0) for summary in entry.regimes.values())
            for entry in stats.scenarios
        ]
        figure.add_trace(
            go.Bar(
                x=names,
                y=counts,
                name=kind,
                marker_color=palette[index % len(palette)],
                hovertemplate="%{x}<br>" + kind + " × %{y}<extra></extra>",
            )
        )
    figure.update_layout(
        **layout(
            "Refusals by reason — which layer caught what",
            "",
            "Cycles refused",
            height=360,
        ),
        barmode="stack",
    )
    return figure
