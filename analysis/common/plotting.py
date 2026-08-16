"""Threshold-annotation helpers so a figure carries its own verdict.

The standing convention (``analysis/CLAUDE.md``): an analysis plot is not a bare
curve the reader must interpret. Requirement thresholds are drawn on the axes,
the measured value is annotated *at the point it was measured*, and the verdict
appears as text — never as colour alone, so the figure survives grayscale
printing and colour-blind readers.

The ``Agg`` backend is selected on import: these run under pytest and in CI
where there is no display, and an interactive backend there is a hang rather
than an error.

References
----------
Design doc §12 (analysis tools), §21.2 (generated artifacts).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - backend must be set before pyplot

__all__ = [
    "FAIL_COLOR",
    "NEUTRAL_COLOR",
    "PASS_COLOR",
    "THRESHOLD_COLOR",
    "annotate_measurement",
    "plt",
    "save",
    "threshold_line",
    "verdict_color",
    "verdict_title",
]

#: Verdict colours. Paired with text in every helper below, never load-bearing
#: on their own.
PASS_COLOR = "#1a7f37"
FAIL_COLOR = "#b3261e"
THRESHOLD_COLOR = "#b3261e"

#: For a measurement with no threshold to judge it against. Deliberately not a
#: verdict colour: annotating an unjudged value in green would read as a pass.
NEUTRAL_COLOR = "#1f4e79"


def verdict_color(passed: bool) -> str:
    """Colour for a pass/fail verdict.

    Parameters
    ----------
    passed : bool

    Returns
    -------
    str
        A matplotlib colour string.
    """
    return PASS_COLOR if passed else FAIL_COLOR


def threshold_line(
    ax, value: float, label: str, *, orientation: str = "h", **kwargs
) -> None:
    """Draw a requirement threshold across an axis, labelled.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    value : float
        Threshold value, in the axis's units.
    label : str
        Legend label, e.g. ``"GM req 6 dB"``.
    orientation : {'h', 'v'}, optional
        Horizontal (``axhline``) or vertical (``axvline``).
    **kwargs
        Passed through to the matplotlib call.
    """
    style = {"color": THRESHOLD_COLOR, "ls": "--", "lw": 1.2, "label": label}
    style.update(kwargs)
    (ax.axhline if orientation == "h" else ax.axvline)(value, **style)


def annotate_measurement(
    ax, x: float, y: float, text: str, passed: bool | None = None, *, offset=(8, 8)
) -> None:
    """Mark a measured value at the point it was measured, with its verdict.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : float
        Data coordinates of the measurement.
    text : str
        The value, already formatted with units.
    passed : bool or None, optional
        Whether this measurement satisfies its requirement; drives the colour
        **and** the appended verdict word. Pass ``None`` — the default — for a
        measurement that **has no threshold to be judged against**: the value is
        annotated in a neutral colour with no verdict word. A PASS that cannot
        fail is not a verdict, so the convention is that PASS/FAIL appears only
        where a criterion exists.
    offset : tuple of int, optional
        Annotation offset in points.
    """
    color = NEUTRAL_COLOR if passed is None else verdict_color(passed)
    ax.plot([x], [y], "o", color=color, markersize=6, zorder=5)
    ax.annotate(
        text if passed is None else f"{text}  {'PASS' if passed else 'FAIL'}",
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=8,
        color=color,
        zorder=6,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.85},
    )


def verdict_title(ax, title: str, passed: bool, subtitle: str = "") -> None:
    """Set a title that states the verdict in words as well as colour.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    title : str
        Main title — by convention it carries the configuration name.
    passed : bool
        Overall verdict for what the figure shows.
    subtitle : str, optional
        Second line, e.g. the analysis conditions.
    """
    word = "PASS" if passed else "FAIL"
    text = f"{title} — {word}"
    if subtitle:
        text += f"\n{subtitle}"
    ax.set_title(text, color=verdict_color(passed), fontsize=10)


def save(fig, path: str | Path, dpi: int = 120) -> Path:
    """Write a figure and close it, creating parent directories.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    path : str or pathlib.Path
        Destination file.
    dpi : int, optional
        Output resolution.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target
