# CLAUDE.md — `analysis/` (Python Analysis Tools)

> **Status: one package implemented, the rest planned.** `control/` is live as of
> Push 55 — the linear control-analysis toolkit (§8.5, §13): stability margins,
> controllability and observability of the as-flown configuration, with tests in
> `tests/analysis/`. Every other subdirectory (`momentum/`, `detumble/`,
> `contacts/`, `linkbudget/`, `postproc/`) is still a placeholder and waits on
> `bindings/` exposing the C++ it must reuse (§13/§21.4, Phase 11).

Python tools for mission analysis: RW/CMG momentum budgeting & sizing, detumble-time MC, ground-station contact scheduling, link budget, pointing budgets, post-processing. Read the root `CLAUDE.md` first.

Dependencies live in the **`analysis` group** of `pyproject.toml` (`uv sync --group analysis`), deliberately not a default group: this is a ground-side lane, not build tooling. It is not optional for the test suite, though — the margin requirement REQ-ACTL-006 is verified by `uv run --group analysis pytest`, and CI runs it that way. The group is **numpy, scipy and matplotlib** (plus the config compiler's pydantic/pyyaml). Adding a domain library needs a reason bigger than convenience: `analysis/control/` owns its margin extraction and Gramians in ~200 lines of numpy because the library answer for a conditionally stable loop is a number the requirement cannot be written on anyway. What owning it costs is the obligation to prove it — see the closed-form validation cases in `tests/analysis/test_control_margins.py`.

## Every analysis reports and plots its own verdict (standing convention)

An analysis with a pass/fail criterion produces **both**, always. `analysis/common/` holds the shared pieces so this is a pattern and not one push's habit; reuse it rather than reinventing it.

1. **A report.** `analysis.common.report.AnalysisReport` — a list of `Criterion` (requirement ID, threshold, measured value, units, sense, note), each computing its own margin in absolute and percentage terms, plus the **configuration provenance** (which YAML, which gains, which mode) and the **modelling assumptions in force**. A margin without its assumptions is not a result, and a margin that is only printed is a number rather than a property. `format_text()`/`write_text()` render it beside the plots; the structured object is what tests assert on. **Never parse rendered text or images for a verdict.**
2. **Plots that show the verdict on their face.** Not bare curves the reader must interpret. Draw the requirement threshold on the axes (`analysis.common.plotting.threshold_line`), annotate each measured value *at the point it was measured* with its number and the word PASS or FAIL (`annotate_measurement`), and put the config name and overall verdict in the title (`verdict_title`). **Colour is never load-bearing on its own** — every verdict is also words, so the figure survives grayscale and colour-blind readers.

Output goes to a caller-supplied directory, defaulting under `build-artifacts/`; figures and rendered reports are derived artifacts and are never committed.

**State the validity boundary, and check it where checking is cheap.** `analysis/control/` is per-axis SISO, which assumes a near-diagonal inertia and small stored wheel momentum; it refuses a non-diagonal tensor at load and *measures* the gyroscopic coupling ratio at the loop crossover, emitting a report warning when the assumption is questionable for the configuration analysed (it fires on the reference vehicle at full wheel momentum). Warnings qualify a report; they do not fail it. See design doc §8.5, "SISO validity boundary and MIMO roadmap".

## The one rule that matters most

**Reuse the flight/sim C++ via `bindings/` (pybind11). Do not reimplement GNC math in Python.** The whole point is that analysis exercises the *same* code that flies — propagation, frames, time, quaternions, momentum allocation, link geometry all come through the bindings. If a needed function isn't bound yet, **add the binding** (or ask the **gnc-algorithms**/**fsw-fprime** agents to expose it), don't write a parallel NumPy version that will silently drift.

Exceptions: plotting, data wrangling, statistics, scheduling glue, and report generation are legitimately Python-only.

**And one exception with a boundary worth stating: `control/`.** Linear-systems analysis — transfer functions, Gramians, rank tests, frequency responses — operates on *matrices read from the committed `config/` YAML*, not on states the flight software computes. There is no flight implementation of a Bode plot to drift from; what the package analyses is the flight **design**, and it reads that design from the same file the vehicle does. The line: it computes no quaternion, no frame transform, no propagation and no filter update. The moment a tool here needs one of those it needs a binding, not a NumPy copy — and if the binding does not exist yet, say so and stop rather than writing the copy.

## Conventions

- **numpydoc** docstrings (NumPy style) on everything public — these auto-compile into the Sphinx site. State **units and frames** and cite the **reference** (`docs/refs.bib`) for any algorithm.
- SI internally; convert to friendly units (km, deg, dB) only at the presentation/report boundary.
- Frame-tagged quantities: respect the frame the binding returns; don't strip frames into bare arrays and lose track.
- Configs come from the **config compiler** outputs / the same `config/` YAML — don't hard-code spacecraft parameters in scripts.
- Pinned, reproducible environment. Deterministic: seed any MC explicitly so results reproduce.

## Tooling

- `pytest` for tests; compare analytic/known cases and (where relevant) against the same **GMAT golden** data the C++ uses.
- These tools also become the backend for the future **live web tools** (RW momentum budget, etc.) — keep them importable and side-effect-free so they can run via WASM/Pyodide or a thin backend on the same code.
