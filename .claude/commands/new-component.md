---
description: Scaffold a new F´ GNC component following the Polaris flight standard
argument-hint: <ComponentName> [one-line purpose]
---

Scaffold a new F´ component named **$1** for Polaris. Purpose: $ARGUMENTS

Follow `flight/CLAUDE.md` and design-doc §4. Delegate to the **fsw-fprime** subagent. Produce:

1. An **FPP model** defining the component with:
   - typed input/output **ports** (consuming `EstimatedState` where relevant — never `TruthState`),
   - **commands**, **telemetry channels** (with declared SI units and frames), **events (EVRs)** with sensible severities, and **parameters** (loaded from the config compiler / `ParameterDb`, no hard-coded physical values),
   - placement in the appropriate **rate group** (state the rate).
2. Generated implementation stubs honoring the flight model: **no heap after init, fixed-size Eigen only, no exceptions, no recursion, return codes checked**.
3. **Comprehensive health telemetry** (state, validity, mode, margins) for the component.
4. A **component unit test** skeleton.
5. Docstrings with **units, frames, and a `refs.bib` reference** for any embedded algorithm.
6. A note on which **`REQ-###`** this component traces to (flag if none exists yet).

Do not wire physical/model parameters to defaults — there are none; reference config. When done, summarize the ports/commands/channels/events created and what remains to wire into the topology, then recommend running **fsw-code-reviewer** on the diff.
