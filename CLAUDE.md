# CLAUDE.md — Polaris GNC FSW

Polaris is a from-scratch spacecraft GNC software suite:
- **`flight/`** — flight software in **F´ (F Prime) / C++**, flight-grade, NASA-standard.
- **`sim/`** — high-fidelity **6DOF truth/environment simulation** (C++), the "plant" the FSW runs against (SITL).
- **`lib/`** — shared C++ used by both (math, frames, time, state, environment, ephemeris, models).
- **`analysis/`** — Python analysis tools, calling the *same* C++ via **`bindings/` (pybind11)**.
- **`mc/`** — Monte Carlo framework.

**The authoritative spec is `docs/design/Polaris_Design_Document.md`.** Read the relevant section before any non-trivial work. This file is the operational summary, not a replacement.

---

## Golden rules — never violate without explicit human sign-off

1. **Time:** onboard master clock is **TAI** (monotonic int64 ns). GNSS reports **GPS time + ECEF** → convert `TAI = GPS + 19 s` and ECEF→ECI on ingest. UTC is derived for ground only. Never run onboard logic in UTC.
2. **Attitude:** quaternions are **JPL convention, scalar-first** `[q0, q1, q2, q3]`, `q0` scalar, canonical `q0 ≥ 0`. One quaternion library; no ad-hoc quaternion math.
3. **Units:** **SI everywhere** inside FSW and lib. Display-unit conversion (deg, km) happens only at the ground/analysis boundary.
4. **Frames:** every vector/quaternion is **frame-tagged**. At module/port/state boundaries use the **boundary typed-vector wrappers** (`Vec3<ECI>`, `Quat<Body,ECI>`, …) — mixing frames across a boundary must be a *compile error*. Raw fixed-size Eigen is allowed only inside hot kernels, re-tagged on the way out. All rotations route through the single transform library.
5. **Canonical state:** there is exactly one nav product — **`EstimatedState`** (onboard) and **`TruthState`** (sim). Everything consumes these; no per-component ad-hoc state. The FSW must be structurally unable to read `TruthState` (truth never leaks into flight).
6. **Flight memory model:** in `flight/` and flight paths of `lib/`, **no dynamic memory allocation after init**, **fixed-size Eigen only** (dynamic-size types banned), no recursion, bounded loops, **no C++ exceptions**. Faults → F´ events → FDIR action and/or operator alert; every alert maps to an action or an explicit no-action.
7. **Reference provenance:** every algorithm/model/numerical method cites its **source (textbook chapter preferred, else paper)** in the header/docstring, keyed to `docs/refs.bib`. No unsourced "magic" formulas.
8. **Config is the single source of truth:** spacecraft/scenario YAML + hardware-library model-IDs are compiled by the **config compiler** into F´ params, sim setup, and analysis inputs. Don't hand-edit derived params; edit config and recompile. On-orbit uplinks are a versioned overlay on the authoritative ground baseline.
9. **Verification is sourced too:** numerical functions (propagation, conversions, frames, eclipse, contacts) are validated against **GMAT golden fixtures** (`tests/golden/`). **Do not use Orekit.**

If a task seems to require breaking one of these, **stop and ask** — these are design decisions (`§18` of the design doc), not defaults.

---

## Architecture invariants

- **SITL is two processes** (`sim` + `flight`) over **F´ TCP byte-stream transport** (`Drv::TcpServer/Client` + `Svc::Framing`), localhost. This link is *test infrastructure*, not a flight bus — don't put flight fault-tolerance logic in it. The flight-representative HW boundary is the F´ `Drv` layer (maps to SpaceWire/1553/CAN/RS-422 on a real target).
- **Execution is sim-time-driven and lockstepped**, never wall-clock-driven. The sim owns the master clock; per macro-step it integrates → publishes due sensor samples → FSW rate groups fire (10 Hz control default) → FSW emits actuator commands → sim applies next step. Reproducible bit-for-bit from `{config, seed}` at any speed.
- **Truth vs onboard:** truth models and onboard models *deliberately differ* (fidelity, biases, noise, latency). Don't "fix" the estimator by feeding it truth — that defeats the sim.
- **Actuators are modular:** a vehicle is **RW-or-CMG, not both**. Guidance/control emit a commanded body torque; the allocation layer is the swappable piece.

---

## Repo layout

```
flight/      F´ deployment: components/ ports/ topology/ config/   (NO HEAP, NO EXCEPTIONS)
sim/         truth plant: dynamics(RK89) world/ io(F´ TCP + sim-time sync) + fault injection
lib/         shared C++: math(Eigen,quat,typed-vec) frames(+EOP) time(TAI/UTC/GPS)
             state/(EstimatedState/TruthState) constants/(registry incl WGS84)
             environment(grav,drag,SRP,IGRF,3body) ephemeris(SPICE ground + Cheby onboard) models/
bindings/    pybind11 exposing lib/ to Python (and future WASM web tools)
analysis/    momentum/ detumble/ contacts/ linkbudget/ postproc/
config/      hardware/(model library)  spacecraft/  scenarios/
tests/       unit/ component/ integration/ regression/ golden/(GMAT fixtures)
tools/       gmat/(golden-data gen)  configc/(config compiler)  dev/CI scripts
mc/          Monte Carlo campaign configs + runners
docs/        design/ icd/ requirements/  + Sphinx site, refs.bib, Doxygen config
```

Per-directory `CLAUDE.md` files exist in `flight/`, `sim/`, and `analysis/` — they carry the rules specific to that layer and override nothing here, only add.

---

## Build / test / lint (intended commands — wire up in Phase 0)

> The project is scaffolded in Phase 0; use these conventions from the start.

- **Build (C++/F´):** CMake + `fprime-util build`. Host targets: WSL/Linux + macOS. Warnings-as-errors.
- **Unit/component tests:** GoogleTest via CTest (`ctest`); Python tests via `pytest`.
- **GMAT golden regression:** `tools/gmat/` regenerates fixtures; tests compare within documented per-quantity tolerances.
- **Static analysis (must pass):** `clang-format` (enforced), `clang-tidy`, `cppcheck`, ASan/UBSan in test builds.
- **Docs:** unified Sphinx site (numpydoc + Breathe/Doxygen + sphinxcontrib-bibtex). Doc build is a CI gate — broken docstrings or missing references fail.
- **CI:** GitHub Actions: build → static analysis → unit/component → integration SITL → MC smoke → GMAT-golden → coverage (thresholds) → docs.

---

## Coding conventions (quick reference)

- C++: follow **JPL "Power of Ten"** + the **JPL Institutional Coding Standard**. `const`-correct, check every return code, defensive finiteness/range checks on estimator/control outputs.
- Docstrings: **numpydoc** style (Python), structured **Doxygen** blocks (C++). Always state **units and frames** for physical quantities, plus the **reference**.
- Naming: vectors carry their frame (type or name); telemetry channels declare units.
- New flight functionality = new/edited **F´ component** with commands, telemetry channels, events, and parameters — not loose functions.
- New algorithm = implementation **+ reference in `refs.bib` + unit test (ideally GMAT-validated) + traced requirement**.

---

## Workflow (solo developer)

- Feature branches; CI must be green before self-merge to `main` (no review gate, solo repo).
- Pre-commit hooks: format + quick lint. Conventional-commit messages. SemVer + tagged releases; maintain CHANGELOG.
- **Requirements traceability:** every capability maps to a `REQ-###` in `docs/requirements/` and to ≥1 test. Aim for high coverage and **all REQs met with margin** (record method, result, margin).

---

## Development phasing (build order)

Work in dependency order (design doc §24). Don't start a phase whose prerequisites aren't green.

0. Foundations: conventions, typed vectors, **canonical state structs**, math/frames/time/ephemeris libs, config compiler, repo+CI skeleton, **Requirements ICD**, docs site, GMAT harness, license.
1. Truth sim core (6DOF+RK89, gravity/drag/SRP/IGRF/3-body, eclipse, disturbance torques) — GMAT-validated.
2. Sensor & actuator models + hardware library.
3. FSW skeleton (F´) + two-process SITL + onboard time/EOP/Chebyshev/persistence.
4. Attitude determination (initializers, MEKF fine + coarse SS+MAG+IMU, fusion, validity/occlusion).
5. Attitude control (B-dot, PID, RW L-norm/L-inf or CMG steering, momentum mgmt + desat).
6. Orbit determination & propagation (GNSS-sim, MEKF OD + covariance, multi-object, batch LS, SGP4, OEM).
7. Guidance + full mode state machine.
8. Maneuvering + interop exports (CCSDS/TLE/STK/FreeFlyer).
9. Subsystems (power, thermal, comms/link budget).
10. FDIR + fault-injection integration suite.
11. Monte Carlo + analysis tools.
12. Advanced/future (CMG refinements, main prop, slosh/flex, RPOD, live web tools).

---

## Subagents & commands

Specialized subagents live in `.claude/agents/` — delegate to keep context lean:
- **gnc-algorithms** — estimation/guidance/control/dynamics math (conventions + provenance).
- **fsw-fprime** — F´ components/ports/topology, flight memory/exception rules.
- **sim-environment** — truth sim, environment & sensor/actuator truth models, fault injection.
- **test-vv** — unit/integration/MC tests, GMAT golden data, traceability.
- **docs-scribe** — docstrings, refs.bib provenance, Sphinx site.
- **fsw-code-reviewer** — read-only audit of diffs against the Polaris standard (run after writing flight code).

Custom commands in `.claude/commands/`: `/new-component`, `/verify-golden`, `/conventions-check`, `/phase-status`.

---

## When to STOP and ask the human

- Any change to a Golden Rule or a `§18` design decision.
- Introducing a new third-party dependency (license + flight-suitability + export implications).
- Changing a versioned interface/schema (canonical state, config schema, IPC ICD, data products).
- Anything that would let truth state reach the FSW, add heap/exceptions to flight paths, or bypass the config compiler.
- Picking default physical/model parameters — there are **no defaults**; values come from config.
