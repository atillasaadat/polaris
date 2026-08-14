"""The palette and furniture every interactive report figure shares.

A figure on a report page has to agree with the page, and with the other
figures. The page commits to a light ground unconditionally
(:mod:`analysis.common.report_html` declares it in the head, so a browser's
auto-dark-mode cannot re-render it inverted), which means every figure must
state its own background, grid and font colours rather than inheriting whatever
the reader's OS setting would supply. Stating them once, here, is what keeps two
figures on one page from disagreeing about what white is.

Colour is never load-bearing on its own
---------------------------------------
The standing convention in ``analysis/CLAUDE.md``: every verdict a figure draws
is also the *word* PASS or FAIL, in the label and in the hover text. The verdict
pair below is chosen against this page's white and checked to stay separable
under protanopia and deuteranopia, where a green/amber pairing is not — but that
separability is a courtesy, not the mechanism. A reader who sees no colour at
all must still get the verdict.

Nothing here computes anything. It is furniture.

References
----------
Design doc §12 (analysis tools), §21.2 (generated artifacts);
``analysis/CLAUDE.md`` (the plotting convention).
"""

from __future__ import annotations

__all__ = [
    "AXIS_COLOR",
    "FAIL_COLOR",
    "FONT_COLOR",
    "GRID_COLOR",
    "MUTED_COLOR",
    "PAPER_BG",
    "PASS_COLOR",
    "PLOT_BG",
    "SERIES_1",
    "SERIES_2",
    "SERIES_3",
    "STRUCTURE_COLOR",
    "ZERO_COLOR",
    "layout",
]

#: Verdict colours. Reserved for a *verdict* — a bar that is a PASS or a FAIL —
#: never for a category and never for a reference line, so that seeing one of
#: these on a page means a judgement was made.
PASS_COLOR = "#0d5226"
FAIL_COLOR = "#e5484d"

#: The categorical series, assigned in a fixed order and never cycled: series 1
#: is always the first category a figure introduces. Figures label their series
#: directly as well, so the colour aids grouping rather than keying the chart.
SERIES_1 = "#0969da"
SERIES_2 = "#bc4c00"
SERIES_3 = "#6a3d9a"

#: Structure, not data: the deep slate the page uses for its rules and table
#: heads, for the frame a figure's data sits in rather than for the data.
STRUCTURE_COLOR = "#24364a"

#: A third-rank series that should recede behind the two being compared.
MUTED_COLOR = "#8a8f98"

#: Light-theme figure furniture, matching the page's own ground.
PAPER_BG = "#fcfcfb"
PLOT_BG = "#ffffff"
GRID_COLOR = "#e4e2dd"
ZERO_COLOR = "#c9c6c0"
FONT_COLOR = "#1a1d21"
AXIS_COLOR = "#5b6470"


def layout(
    title: str,
    x_title: str,
    y_title: str,
    *,
    height: int = 420,
    log_y: bool = False,
) -> dict:
    """Standard 2D layout for a report figure.

    Parameters
    ----------
    title : str
        Figure title. Short: the caption beneath carries the sentence.
    x_title, y_title : str
        Axis labels, including units. A report figure always states its units on
        the axis — a caption can be scrolled past, an axis cannot.
    height : int, optional
        Pixels.
    log_y : bool, optional
        Log the vertical axis. For an error that spans orders of magnitude a
        linear axis shows one regime and hides the rest.

    Returns
    -------
    dict
        Ready to pass to ``go.Figure.update_layout``.
    """
    axis = {
        "gridcolor": GRID_COLOR,
        "zerolinecolor": ZERO_COLOR,
        "color": AXIS_COLOR,
        "linecolor": AXIS_COLOR,
    }
    return {
        "title": {"text": title, "font": {"size": 13, "color": FONT_COLOR}},
        "font": {"color": FONT_COLOR, "size": 11},
        "xaxis": {**axis, "title": x_title},
        "yaxis": {**axis, "title": y_title, "type": "log" if log_y else "linear"},
        "margin": {"l": 70, "r": 20, "t": 44, "b": 52},
        "height": height,
        "legend": {"orientation": "h", "y": -0.18, "font": {"size": 10}},
        "paper_bgcolor": PAPER_BG,
        "plot_bgcolor": PLOT_BG,
        "hovermode": "closest",
    }
