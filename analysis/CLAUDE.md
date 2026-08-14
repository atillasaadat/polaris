# CLAUDE.md — `analysis/` (Python Analysis Tools)

> **Status: one package implemented, the rest planned.** `control/` is live as of
> Push 55 — the linear control-analysis toolkit (§8.5, §13): stability margins,
> controllability and observability of the as-flown configuration, with tests in
> `tests/analysis/`. Push 56 added its **command-line gate**,
> `PYTHONPATH=tools uv run --group analysis python -m analysis.control config/spacecraft/leo_smallsat.yaml`,
> which writes the figures and the rendered report, prints the report and **exits
> non-zero on any FAIL** — the pre-simulation design check (`--no-plots` skips the
> figures, `--out` chooses the directory). Warnings qualify a report and never
> fail it, per the convention below.
>
> **`detumble/` is live as of Push 59** — the B-dot residual-spin-tail Monte Carlo
> (§13, §23.2; REQ-ACTL-001's recorded owed item). It is the first package built
> on the campaign pattern: the **runs are C++** (`tests/mc/detumble_mc.cpp` →
> `polaris_detumble_mc`, which forks the real deployment and drives the real
> `ClosedLoop`, so no GNC math is reimplemented), and this package reads the
> JSONL the driver writes and owns the sampling statistics, the figures and the
> report. `PYTHONPATH=tools uv run --group analysis python -m analysis.detumble <records>` prints
> the report and exits non-zero on any FAIL; see `analysis/detumble/README.md`
> for how to fly the campaign. It needs no bindings precisely because the runs
> stay on the C++ side — which is the shape any future campaign package should
> copy.
>
> **`sizing/` is live as of Push 60** — ADCS actuator sizing and design
> validation (§12, §7, §8.5). It **fulfils the `momentum/` placeholder** and is
> deliberately broader than momentum budgeting: it computes the actuator
> envelopes (the exact zonotope, its inscribed radius, the L2 ellipsoid inside
> it), validates a candidate design against its sizing drivers with 30 % margin,
> and **derives the flight tuning that follows from the design** — PID gains,
> momentum envelope, desaturation hysteresis, detumble exit threshold, B-dot
> gain — each returned with its formula, its inputs and its reasoning.
> `PYTHONPATH=tools uv run --group analysis python -m analysis.sizing config/spacecraft/leo_smallsat.yaml`
> writes an **interactive HTML report** to `<out>/index.html`, opens it in a
> browser, and exits non-zero on any FAIL (`--no-browser` to write without
> opening — what CI and the tests use; `--print` to also get the console report,
> which is unchanged in content and still written to `<out>/sizing_report.txt`
> either way, `--no-plots` included, since the text rendering is the record and
> not a figure). It runs on a config **anywhere on disk**: the hardware catalog
> resolves to the sibling `config/hardware` when there is one and to this
> repository's otherwise, located relative to the package rather than the working
> directory. The page is the default because the headline result is a 3D
> achievable set with the certified ceiling nested inside the hardware one, and
> that is a shape a reviewer has to rotate rather than a table they can read.
> It needs no bindings for the
> same reason `control/` does not: it reads matrices and scalars from the
> committed config and computes no quaternion, frame or propagation. It reuses
> `control/`'s vehicle loader, its `siso_coupling` momentum boundary, its
> `axis_margins` crossover and its dipole field model rather than re-deriving
> any of them, and parses the atmosphere band table out of
> `sim/world/atmosphere.cpp` rather than transcribing it. See
> `analysis/sizing/README.md`.
>
> **`od/` is live as of Push 63** — the 7-day orbit-determination Monte Carlo
> (§8.3, §9.2, §13, §23.2). Same shape as `detumble/`: the runs are C++
> (`tests/mc/orbit_od_mc.cpp` → `polaris_orbit_od_mc`, driving the real
> `gnc::OrbitOd` against the real receiver model), and this package owns the
> statistics, the figures and the verdict.
> `PYTHONPATH=tools uv run --group analysis python -m analysis.od <records>`
> writes the interactive page, prints the report and exits non-zero on any FAIL.
> Its centre of gravity is that a covariance can be wrong in three separable
> ways, so it carries three checks: NEES and NIS (self-normalised, in
> `statistics.py`) and the **ensemble spread against the reported σ per RIC
> axis** (`ensemble.py`), which estimates the covariance a second time from
> truth alone and is the only one of the three that can see an error and a
> covariance wrong by the same factor. Two more live on the C++ side and need no
> campaign — `tests/unit/orbit_od_covariance_test.cpp` bounds the propagated
> covariance against a finite-difference STM and against Liouville volume
> conservation, both with no reference implementation. See
> `analysis/od/README.md`.
>
> `contacts/`, `linkbudget/` and `postproc/` are still placeholders and wait on
> `bindings/` exposing the C++ they must reuse (§13/§21.4, Phase 11).

Python tools for mission analysis: RW/CMG momentum budgeting & sizing, detumble-time MC, ground-station contact scheduling, link budget, pointing budgets, post-processing. Read the root `CLAUDE.md` first.

Dependencies live in the **`analysis` group** of `pyproject.toml` (`uv sync --group analysis`), deliberately not a default group: this is a ground-side lane, not build tooling. It is not optional for the test suite, though — the margin requirement REQ-ACTL-006 is verified by `uv run --group analysis pytest`, and CI runs it that way. The group is **numpy, scipy, matplotlib and plotly** (plus the config compiler's pydantic/pyyaml). Adding a domain library needs a reason bigger than convenience: `analysis/control/` owns its margin extraction and Gramians in ~200 lines of numpy because the library answer for a conditionally stable loop is a number the requirement cannot be written on anyway. What owning it costs is the obligation to prove it — see the closed-form validation cases in `tests/analysis/test_control_margins.py`.

**plotly is the one library admitted on the opposite argument, and the boundary is that it renders nothing but what numpy computed.** `analysis/sizing/` reports a three-dimensional achievable set with a second surface nested inside it, and the reader has to rotate that to believe it; hand-rolling an interactive WebGL viewer is far more code than the dependency costs, and unlike a margin algorithm there is no correctness claim to own — a wrong picture of a right number is a rendering bug, not a wrong answer. It is emitted with `include_plotlyjs="inline"` so the report is one self-contained file that works offline and fetches nothing at runtime, and it stays on the ground-side lane: no flight or sim code imports it. The rule this keeps intact is the one above it — **the structured report object is still the verdict**, and the tests assert on that, never on the page.

**KaTeX is admitted on the same argument and vendored rather than depended on.** The page typesets its formulae, and the alternatives were a lookup table feeding matplotlib mathtext (which is what Push 60 shipped, and which rotted the moment a formula string was reworded) or hand-rolled Unicode that cannot set a fraction. KaTeX 0.16.11 is committed **verbatim** under `analysis/common/vendor/katex/` with a `PROVENANCE.md` recording the version, the jsdelivr URLs and the MIT licence — the same convention §3.7 sets for external reference data, and for the same reason: the page inlines it, so the committed bytes must be reproducible from their source. It is not a Python dependency and adds nothing to the `analysis` group. The boundary is the same one plotly is held to: **it renders, it never computes.** Every symbol it sets comes from a `formula_tex` written beside the ASCII formula on the object that owns it, so a formula edited upstream carries its LaTeX with it, and one without a LaTeX form renders as plain text rather than as a guess.

**`PYTHONPATH=tools` is not optional on these commands.** Every package here resolves its vehicle through
`configc` (§19.3), which lives in `tools/` and is not an installed distribution — `pytest.ini` puts it on the
path for the test lane, so the suites pass while a hand-run CLI raises `ModuleNotFoundError: configc`. The same
prefix `tools/freeflyer/README.md` documents applies here.

## Every analysis reports and plots its own verdict (standing convention)

An analysis with a pass/fail criterion produces **both**, always. `analysis/common/` holds the shared pieces so this is a pattern and not one push's habit; reuse it rather than reinventing it.

1. **A report.** `analysis.common.report.AnalysisReport` — a list of `Criterion` (requirement ID, threshold, measured value, units, sense, note), each computing its own margin in absolute and percentage terms, plus the **configuration provenance** (which YAML, which gains, which mode) and the **modelling assumptions in force**. A margin without its assumptions is not a result, and a margin that is only printed is a number rather than a property. `format_text()`/`write_text()` render it beside the plots; the structured object is what tests assert on. **Never parse rendered text or images for a verdict.**
2. **Plots that show the verdict on their face.** Not bare curves the reader must interpret. Draw the requirement threshold on the axes (`analysis.common.plotting.threshold_line`), annotate each measured value *at the point it was measured* with its number and the word PASS or FAIL (`annotate_measurement`), and put the config name and overall verdict in the title (`verdict_title`). **Colour is never load-bearing on its own** — every verdict is also words, so the figure survives grayscale and colour-blind readers.

Output goes to a caller-supplied directory, defaulting under `build-artifacts/`; figures and rendered reports are derived artifacts and are never committed.

3. **An HTML rendering, when the headline is a shape rather than a table.** `analysis/common/report_html.py` is the shared design system — one stylesheet, one behaviour script, the KaTeX and plotly plumbing, the prose helpers and the page shell (`render_page`). A report package owns only its *sections*: which exist, in what order, and what goes in each. Reuse the shell; never fork it. `analysis/sizing/html.py` and `analysis/od/html.py` are the worked examples, and they share no content.

**Opening a browser is opt-out twice over.** Every report CLI opens its page by default, because a person running it wants to read it — and every one takes `--no-browser` to write without opening, which is what CI and the tests use. `POLARIS_NO_BROWSER` set in the environment overrides *every* caller at once and can never be overridden back on; it is the switch to set when regenerating a page repeatedly, so nothing opens a window on someone's behalf twenty times in a row. The path is printed either way, so a save-only run still says where the artifact landed.

**State the validity boundary, and check it where checking is cheap.** `analysis/control/` is per-axis SISO, which assumes a near-diagonal inertia and small stored wheel momentum; it refuses a non-diagonal tensor at load and *measures* the gyroscopic coupling ratio at the loop crossover, emitting a report warning when the assumption is questionable for the configuration analysed (it fires on the reference vehicle at full wheel momentum). Warnings qualify a report; they do not fail it. See design doc §8.5, "SISO validity boundary and MIMO roadmap".

## The one rule that matters most

**Reuse the flight/sim C++ via `bindings/` (pybind11). Do not reimplement GNC math in Python.** The whole point is that analysis exercises the *same* code that flies — propagation, frames, time, quaternions, momentum allocation, link geometry all come through the bindings. If a needed function isn't bound yet, **add the binding** (or ask the **gnc-algorithms**/**fsw-fprime** agents to expose it), don't write a parallel NumPy version that will silently drift.

Exceptions: plotting, data wrangling, statistics, scheduling glue, and report generation are legitimately Python-only.

**And one exception with a boundary worth stating: `control/` and `sizing/`.** Linear-systems analysis — transfer functions, Gramians, rank tests, frequency responses — operates on *matrices read from the committed `config/` YAML*, not on states the flight software computes. There is no flight implementation of a Bode plot to drift from; what the package analyses is the flight **design**, and it reads that design from the same file the vehicle does. The line: it computes no quaternion, no frame transform, no propagation and no filter update. The moment a tool here needs one of those it needs a binding, not a NumPy copy — and if the binding does not exist yet, say so and stop rather than writing the copy.

`sizing/` sits inside the same boundary and is held to the same line: actuator envelope geometry, closed-form disturbance-torque magnitudes and gain placement are all functions of matrices and scalars the config carries. Where it needs something the plant already implements it **reads the plant's own data** rather than restating it — the atmosphere band table is parsed out of `sim/world/atmosphere.cpp`, the wheel capacity and magnetometer noise out of `config/hardware/`, the geomagnetic field out of the committed IAGA table through `control/field` — and where the model is a truncation (degree-1 IGRF, static exponential density) it says so in the report's assumptions rather than in a comment nobody reads.

## Conventions

- **numpydoc** docstrings (NumPy style) on everything public — these auto-compile into the Sphinx site. State **units and frames** and cite the **reference** (`docs/refs.bib`) for any algorithm.
- SI internally; convert to friendly units (km, deg, dB) only at the presentation/report boundary.
- Frame-tagged quantities: respect the frame the binding returns; don't strip frames into bare arrays and lose track.
- Configs come from the **config compiler** outputs / the same `config/` YAML — don't hard-code spacecraft parameters in scripts.
- Pinned, reproducible environment. Deterministic: seed any MC explicitly so results reproduce.

## Tooling

- `pytest` for tests; compare analytic/known cases and (where relevant) against the same **GMAT golden** data the C++ uses.
- These tools also become the backend for the future **live web tools** (RW momentum budget, etc.) — keep them importable and side-effect-free so they can run via WASM/Pyodide or a thin backend on the same code.
