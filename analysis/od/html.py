"""The orbit-OD campaign report as one self-contained, interactive HTML page.

The console rendering (:meth:`analysis.common.report.AnalysisReport.format_text`)
stays the record the tests assert on. This is the second rendering, for the case
the text serves badly — and for a multi-day campaign across seven fault
scenarios (plus the latency case) that case is the normal one: the claim is a shape over time, an error
staying inside a covariance envelope through outages and spoofs, and no table
shows that.

The design system is :mod:`analysis.common.report_html`; this module owns only
which sections exist and what goes in them. That is the same split
:mod:`analysis.sizing.html` is on, and the two share no content.

Nothing here computes a verdict. Every PASS/FAIL word is read from the
:class:`~analysis.common.report.AnalysisReport` handed in.

What the page leads with
------------------------
One sentence: how far from truth the filter ever got, over how long, and whether
its covariance stayed honest. Directly beneath it the error history of the
longest nominal run with its ±3σ envelope, which is that sentence made visible.
The fault scenarios follow one per block, each led by the driver's own stated
intent — quoted from the records, so a scenario whose rationale is reworded takes
the report's wording with it and there is no second copy to go stale.

Why the scenario blocks are not one table
------------------------------------------
Eight scenarios with seven regimes each is fifty-odd rows, and a reader looking
for how the filter handled a six-hour outage should not have to find it among
rows about magnetometer sigmas. Each scenario is its own block with its own
figure, its own regime table and its own refusal breakdown, so the unit of
reading is the question rather than the row.

References
----------
Design doc §8.3, §9.2, §13, §21.2 (generated artifacts); ``analysis/CLAUDE.md``
(the reporting convention).
"""

from __future__ import annotations

import math
from pathlib import Path

from analysis.common.mathfmt import percent, signed, unit_scale
from analysis.common.report import AnalysisReport, Criterion
from analysis.common.report_html import (
    esc,
    figure_block,
    glossary,
    lead_and_why,
    prose,
    render_page,
)
from analysis.od.plots import (
    SCENARIO_TITLES,
    consistency_figure,
    error_history_figure,
    refusal_figure,
    regime_figure,
)
from analysis.od.records import Campaign, ScenarioRun
from analysis.od.statistics import CampaignStatistics, ScenarioStatistics

__all__ = ["write_html"]

#: Section anchors and labels, in document order.
SECTIONS = (
    ("verdict", "Verdict"),
    ("criteria", "Criteria"),
    ("consistency", "Consistency"),
    ("scenarios", "Scenarios"),
    ("assumptions", "Assumptions"),
    ("warnings", "Measurements"),
)

#: The report's own short codes and initialisms, defined in the open at the head
#: of the criteria section — the same convention the sizing report follows for
#: its D/M codes. A reader meeting NEES for the first time cannot expand it from
#: a table row.
ACRONYMS = (
    (
        "NEES",
        "normalised estimation error squared: the state error, squared and "
        "weighted by the inverse of the covariance the filter reported for it. "
        "For an honest filter it averages the state dimension (6).",
    ),
    (
        "NIS",
        "normalised innovation squared: the same construction on the "
        "measurement side, applied to each fix before it is folded in. "
        "Averages the measurement dimension (3), and is what the rejection "
        "gate thresholds.",
    ),
    (
        "Ensemble/reported sigma",
        "the truth-derived standard deviation of the error across independent "
        "runs, divided by the standard deviation the filter claimed. One means "
        "the covariance matches reality; it is the only consistency figure "
        "here that never consults the covariance it is judging.",
    ),
    (
        "Radial, in-track, cross-track",
        "the orbit-local axes (the RIC frame): towards the Earth's centre, "
        "along the velocity, and normal to the orbit plane. Orbit error "
        "concentrates in-track, which is why the ensemble check is resolved "
        "this way rather than as one number.",
    ),
    (
        "Coast horizon",
        "how long the filter may propagate without a fix before its solution "
        "is dropped rather than trusted, 300 s here.",
    ),
    (
        "|dr|/sigma",
        "the position error at an instant, in units of the standard deviation "
        "the filter reported at that instant. Above about 3 the filter is "
        "outside its own error bars.",
    ),
)

#: Regime names as a reader should see them. The record's names are the driver's
#: enum spellings; these say what was happening.
REGIME_LABELS = {
    "nominal": "No fault armed",
    "outage": "Loss of fix",
    "spoof": "Spoofed position",
    "jam": "Geographic jamming",
    "clock_jump": "Receiver clock jump",
    "radius_jump": "Implausible fix (geostationary radius)",
    "sigma_degrade": "Degraded reported sigmas",
}

#: What each refusal name means, said once at the head of the scenarios section
#: rather than repeated under every block that shows one.
REFUSAL_GLOSSARY = (
    ("fix_implausible", "the fix radius is outside the vehicle's orbit band (§9.1)"),
    ("measurement_rejected", "the innovation failed the NIS gate"),
    ("uninitialised", "no solution yet; this entry point cannot seed one"),
    ("coast_expired", "past the coast horizon; the solution was dropped"),
    (
        "non_monotonic_epoch",
        "the epoch went backwards, stalled, or exceeded the latency bound",
    ),
    (
        "step_too_long",
        "the time step is beyond the configured propagation bound: a clock "
        "glitch, not a coast",
    ),
    ("fix_sigma_invalid", "a reported sigma that is not positive and finite"),
    ("fix_not_finite", "a non-finite component in the fix"),
    (
        "frame_conversion",
        "the Earth-fixed to inertial conversion failed, or the epoch is "
        "outside the loaded Earth-orientation data",
    ),
    (
        "no_velocity_for_seed",
        "a seed needs position and velocity both; position alone cannot "
        "start a six-state solution",
    ),
    ("filter_fault", "a non-finite internal result; the solution was dropped"),
)


def _thesis(stats: CampaignStatistics, report: AnalysisReport) -> str:
    """The one sentence the page exists to say, in the words its verdict allows."""
    nominal = stats.of("nominal")
    steady = nominal.regimes.get("nominal") if nominal is not None else None
    days = max((e.duration_s for e in stats.scenarios), default=0.0) / 86400.0
    span = f"{days:.0f} day{'' if round(days) == 1 else 's'}"
    accuracy = (
        f"held <b>{steady.position.p95:.3g} m</b> at the 95th percentile"
        if steady is not None and math.isfinite(steady.position.p95)
        else "was measured"
    )
    honest = ", and its covariance stayed honest" if stats.nees.consistent else ""

    if report.passes:
        return (
            f"Over {span} and {stats.runs} runs the estimate {accuracy}, "
            f"recovered from every fix-stream interruption{honest}."
        )
    failures = report.failures()
    worst = failures[0]
    return (
        f"<b>{len(failures)} of {len(report.criteria)} criteria do not close</b>, "
        f"the first being {esc(worst.name)} at {worst.measured:.4g} against "
        f"{worst.threshold:.4g}. Over {span} and {stats.runs} runs the "
        f"estimate {accuracy}."
    )


def _criteria_table(criteria: tuple[Criterion, ...]) -> str:
    """Every criterion as one sortable table.

    One table and not a grouped set: this report has a dozen criteria, not fifty,
    and grouping a dozen rows costs the reader a scan for no gain.
    """
    rows = []
    for criterion in criteria:
        verdict = "PASS" if criterion.passes else "FAIL"
        scale = unit_scale(criterion.units, [criterion.threshold, criterion.measured])
        rows.append(
            f'<tr class="{"" if criterion.passes else "fail"}">'
            f"<td>{prose(criterion.name)}"
            f'<div class="note">{prose(criterion.note)}</div></td>'
            f'<td class="req">{esc(criterion.requirement)}</td>'
            f'<td class="n">{scale.text(criterion.threshold)}</td>'
            f'<td class="n">{scale.text(criterion.measured)}</td>'
            f'<td class="n">{esc(signed(criterion.margin))}'
            f'<span class="pct">{esc(percent(criterion.margin_pct))}</span></td>'
            f'<td class="n"><span class="verdict {verdict.lower()}">{verdict}</span></td>'
            "</tr>"
        )
    return (
        '<div class="scroll"><table class="sortable"><thead><tr>'
        "<th>Criterion</th><th>Requirement</th><th class=\"n\">Threshold</th>"
        '<th class="n">Measured</th><th class="n">Margin</th><th class="n">Verdict</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _regime_table(entry: ScenarioStatistics) -> str:
    """One scenario's per-regime behaviour."""
    rows = []
    for name, summary in entry.regimes.items():
        refusals = (
            ", ".join(
                f"{kind} × {count}" for kind, count in sorted(summary.refusals.items())
            )
            or "&ndash;"
        )
        rows.append(
            f"<tr><td>{esc(REGIME_LABELS.get(name, name))}</td>"
            f'<td class="n">{summary.position.samples}</td>'
            f'<td class="n">{_metres(summary.position.median)}</td>'
            f'<td class="n">{_metres(summary.position.p95)}</td>'
            f'<td class="n">{_metres(summary.position.worst)}</td>'
            f'<td class="n">{_ratio(summary.sigma_ratio.p95)}</td>'
            f'<td class="n">{summary.accepted} / {summary.fixes}</td>'
            f"<td>{esc(refusals)}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Regime</th>'
        '<th class="n">Samples</th><th class="n">Median</th><th class="n">95th</th>'
        '<th class="n">Worst</th><th class="n">95th |dr|/σ</th>'
        '<th class="n">Fixes used</th><th>Refusals</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _metres(value: float) -> str:
    """A distance, or an em dash where the regime produced no sample."""
    return f"{value:.4g}" if math.isfinite(value) else "&ndash;"


def _ratio(value: float) -> str:
    """A dimensionless ratio, or an em dash."""
    return f"{value:.3g}" if math.isfinite(value) else "&ndash;"


def _longest_run(campaign: Campaign, scenario: str) -> ScenarioRun | None:
    """The run of ``scenario`` with the most samples, for the history figure."""
    runs = campaign.of(scenario)
    return max(runs, key=lambda r: r.samples) if runs else None


def write_html(
    campaign: Campaign,
    stats: CampaignStatistics,
    report: AnalysisReport,
    out_dir: str | Path,
) -> Path:
    """Render the whole campaign result to ``<out_dir>/index.html``.

    Parameters
    ----------
    campaign : analysis.od.records.Campaign
        The loaded records; supplies the per-run traces the figures draw.
    stats : analysis.od.statistics.CampaignStatistics
    report : analysis.common.report.AnalysisReport
        The structured verdict. Every PASS/FAIL word on the page is read from
        here — the page renders the verdict, it never decides it.
    out_dir : str or pathlib.Path
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    verdict = "PASS" if report.passes else "FAIL"

    # The hero is the longest nominal arc: the claim is that the estimate stays
    # inside its covariance for a week, and that is the run where a week of it
    # is actually visible. The fault scenarios each get their own figure below,
    # where the fault is the subject rather than a wrinkle in a long trace.
    hero_run = _longest_run(campaign, "nominal") or (
        campaign.runs[0] if campaign.runs else None
    )
    hero = (
        figure_block(
            error_history_figure(hero_run),
            "The estimate against the covariance it published at the same instant, "
            "over the longest un-faulted arc in the campaign.",
            "Blue is the distance from truth; orange dotted is three times the "
            "filter's own claimed sigma. The claim is containment, not magnitude: "
            "an accurate filter that sits outside its own envelope is overconfident, "
            "and overconfidence is what makes it reject a correct measurement later. "
            "A shaded band is a stretch with no valid solution, which is a different "
            "state from a large error. The vertical axis is logarithmic because a "
            "week spans metres during a fix stream and kilometres during an outage.",
            True,
        )
        if hero_run is not None
        else ""
    )

    provenance = "".join(
        f"<dt>{esc(key)}</dt><dd>{prose(value)}</dd>"
        for key, value in report.provenance.items()
    )
    assumptions = "".join(f"<li>{lead_and_why(a)}</li>" for a in report.assumptions)
    warnings = "".join(f"<li>{lead_and_why(w, 'Detail')}</li>" for w in report.warnings)

    blocks = []
    for entry in stats.scenarios:
        run = _longest_run(campaign, entry.scenario)
        figure = (
            figure_block(
                error_history_figure(run),
                f"{entry.scenario}: the longest run of this scenario, error against "
                "its own covariance.",
                "Shaded bands are stretches with no valid solution. Read them "
                "against the scenario's armed durations: an outage inside the 300 s "
                "coast horizon should leave no band at all, because the solution is "
                "supposed to coast through it.",
                False,
            )
            if run is not None and entry.scenario != "nominal"
            else ""
        )
        blocks.append(
            f'<section class="panel" id="scenario-{esc(entry.scenario)}">'
            f"<h3>{esc(SCENARIO_TITLES.get(entry.scenario, entry.scenario))} "
            f'<span class="code">{esc(entry.scenario)}</span></h3>'
            f'<div class="lead">{prose(entry.intent)}</div>'
            f'<p class="xref">{entry.runs} runs, {entry.samples} cycles at '
            f"{entry.cycle_period_s:g} s"
            + (
                f", fix latency {entry.fix_latency_s * 1e3:g} ms"
                if entry.fix_latency_s > 0
                else ""
            )
            + f", {entry.duration_s / 3600.0:.3g} h each.</p>"
            f"{_regime_table(entry)}"
            f"{figure_block(regime_figure(entry), 'Position error by regime.', 'Quantiles rather than a mean and a standard deviation: the error is bounded below by zero and has a long right tail wherever a fault is armed. Logarithmic, so the nominal stretches and the outage stretches are both readable.', False)}"
            f"{figure}"
            "</section>"
        )

    body = f"""
<section class="hero" id="verdict">
  <div class="statement {verdict.lower()}">
    <span class="pill {verdict.lower()}">{verdict}</span>
    <h1>{_thesis(stats, report)}</h1>
    <span class="tally">{len(report.failures())} failing criteria of {len(report.criteria)}</span>
  </div>
  {hero}
  <div class="spec"><dl>{provenance}</dl></div>
</section>

<section id="criteria"><h2>Criteria</h2>
<p class="lede">No requirement writes a number on this filter's accuracy, so the
accuracy figures are measurements and appear above. What is judged here is what
has a threshold this campaign did not choose: the chi-square consistency
intervals, the gate's own configured rejection rate, and the fault policy.
Click a column heading to sort; failing rows are shaded <b>and</b> say FAIL.</p>
{glossary(ACRONYMS, xref=False)}
{_criteria_table(report.criteria)}
</section>

<section id="consistency"><h2>Consistency</h2>
<p class="lede">The question a worst case cannot answer. A filter can be accurate
and overconfident at once, and the overconfident one is the one that will
eventually refuse a correct measurement. Both series are normalised by their own
degrees of freedom, so a consistent filter sits at 1.0 in either.</p>
{figure_block(consistency_figure(stats), "Measured average normalised error against its chi-square acceptance interval, per scenario.", "The shaded band behind each marker is that scenario's own interval, computed over the number of independent runs it flew. Above the band the covariance is smaller than the error, which is the unsafe direction; below it the filter is merely wasteful. The intervals are computed over run means and not over cycles, because consecutive cycles of one run are correlated over the filter's own time constants and are not independent samples.", False)}
{figure_block(refusal_figure(stats), "Which layer refused what.", "The distinction this chart exists for: an implausible fix refused by the innovation gate looks identical, in any count of accepted against rejected, to one refused at the plausibility band. They are not the same defence — the innovation gate does not exist on the seed path, where the filter has no prior — so the reason is stacked rather than the total.", False)}
</section>

<section id="scenarios"><h2>Scenarios</h2>
<p class="lede">Each scenario, led by the rationale the campaign driver states
for flying it. Regimes are kept apart: a scenario's un-faulted stretches and its
faulted ones are different populations, and their average describes no state the
vehicle is ever in.</p>
<dl class="glossary">{"".join(f"<dt>{esc(name)}</dt><dd>{prose(meaning)}</dd>" for name, meaning in REFUSAL_GLOSSARY)}</dl>
{"".join(blocks)}
</section>

<section id="assumptions"><h2>Assumptions in force</h2>
<p class="lede">A margin without its assumptions is not a result.</p>
<div class="panel"><ul class="block">{assumptions}</ul></div>
</section>

<section id="warnings"><h2>Measurements that qualify this campaign</h2>
<p class="lede">These do not fail the analysis. Several are numbers a future
requirement should be written from rather than results judged against one.</p>
<div class="panel warn"><ul class="block warn">{warnings}</ul></div>
</section>
"""

    page = render_page(
        title=report.title,
        verdict=verdict,
        config_path=report.config_path,
        sections=SECTIONS,
        body=body,
    )
    target = directory / "index.html"
    target.write_text(page, encoding="utf-8")
    return target
