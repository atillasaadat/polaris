# CLAUDE.md — `analysis/` (Python Analysis Tools)

Python tools for mission analysis: RW/CMG momentum budgeting & sizing, detumble-time MC, ground-station contact scheduling, link budget, pointing budgets, post-processing. Read the root `CLAUDE.md` first.

## The one rule that matters most

**Reuse the flight/sim C++ via `bindings/` (pybind11). Do not reimplement GNC math in Python.** The whole point is that analysis exercises the *same* code that flies — propagation, frames, time, quaternions, momentum allocation, link geometry all come through the bindings. If a needed function isn't bound yet, **add the binding** (or ask the **gnc-algorithms**/**fsw-fprime** agents to expose it), don't write a parallel NumPy version that will silently drift.

Exceptions: plotting, data wrangling, statistics, scheduling glue, and report generation are legitimately Python-only.

## Conventions

- **numpydoc** docstrings (NumPy style) on everything public — these auto-compile into the Sphinx site. State **units and frames** and cite the **reference** (`docs/refs.bib`) for any algorithm.
- SI internally; convert to friendly units (km, deg, dB) only at the presentation/report boundary.
- Frame-tagged quantities: respect the frame the binding returns; don't strip frames into bare arrays and lose track.
- Configs come from the **config compiler** outputs / the same `config/` YAML — don't hard-code spacecraft parameters in scripts.
- Pinned, reproducible environment. Deterministic: seed any MC explicitly so results reproduce.

## Tooling

- `pytest` for tests; compare analytic/known cases and (where relevant) against the same **GMAT golden** data the C++ uses.
- These tools also become the backend for the future **live web tools** (RW momentum budget, etc.) — keep them importable and side-effect-free so they can run via WASM/Pyodide or a thin backend on the same code.
