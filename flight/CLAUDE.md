# CLAUDE.md — `flight/` (Flight Software, F´ / C++)

This is **flight-grade** code. The hard constraints here are stricter than the rest of the repo. Read the root `CLAUDE.md` first; this file adds the flight-only rules.

## Hard constraints (non-negotiable in flight paths)

- **No dynamic memory allocation after initialization.** No `new`/`malloc`/`std::vector` growth/`std::string` in steady state. Allocate fixed-size buffers at init. Dynamic-size Eigen types are **banned** — use fixed-size (`Eigen::Matrix<double, N, M>`) only. Estimator/control state is **double-precision** (§4.3) — no single-precision in nav/control numerics, including scalar state and gains.
- **No C++ exceptions** in flight logic. No `throw`, no exception-throwing paths. Surface faults as **F´ events (EVRs)** with the right severity, which drive FDIR responses and/or operator alerts. Every alert maps to a documented action (or explicit no-action).
- **No recursion.** Bounded loops with explicit bounds. No unbounded blocking calls.
- **Check every return code.** No silent failure. Add finiteness/range checks on estimator and control outputs.
- **Determinism & WCET:** high-rate paths must be bounded and deterministic. The 10 Hz control frame has a timing budget (design doc §23.5) — respect it.
- Follow **JPL "Power of Ten"** and the **JPL Institutional Coding Standard**. `const`-correctness throughout.

## F´ structure

- New functionality = a new or modified **F´ component** with explicit **commands, telemetry channels, events, parameters** — not free functions bolted onto a topology.
- Inter-component communication is **only through typed ports**. No back-channels, no globals, no shared mutable state outside ports.
- **Rate groups** are configuration, not hard-coded constants. Control loop default 10 Hz; estimation/OD/housekeeping at their configured rates.
- Hardware access goes through the **`Drv` HAL** components. Keep the GNC↔driver boundary clean typed ports so a future flight-bus swap (SpaceWire/1553/CAN/RS-422) is a driver-layer change only.
- Parameters come from the **config compiler** (`tools/configc/`) into `ParameterDb`. Never hard-code physical values; there are no defaults.

## State, time, frames

- Consume **`EstimatedState`** (`lib/state/`). The FSW must be structurally unable to see `TruthState`.
- **Respect validity flags.** `EstimatedState` carries per-field validity (§8.0) and each sensor measurement has a validity flag — range/rate-of-change/staleness/cross-sensor/solution-quality criteria (§9.1). Invalid or stale data is **excluded, never silently used**; gate every consumer (guidance, control, FDIR, telemetry) on the flag.
- Time is **TAI**. The FSW master clock is monotonic **int64 ns**; use the **two-part high-precision form (int64 s + double frac)** for long-arc work (onboard OD/propagation, Chebyshev ephemeris) per §3.2. Convert GNSS **GPS time + ECEF → TAI + ECI** on ingest (`TAI = GPS + 19 s`, ECEF→ECI via onboard EOP).
- Use **boundary typed vectors** at component/port interfaces; raw fixed-size Eigen only inside kernels.
- Onboard ephemeris is **Chebyshev** fits + uploaded EOP/leap-second tables — **never SPICE onboard** (SPICE is ground/sim only).

## Before you commit flight code

1. Run the **fsw-code-reviewer** subagent on your diff.
2. Confirm: no heap, no exceptions, fixed-size Eigen, return codes checked, units/frames/reference in docstrings.
3. New algorithm → `refs.bib` entry + unit test (GMAT-validated where applicable) + `REQ-###` trace.
