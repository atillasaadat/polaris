---
description: Audit current changes against the Polaris conventions and flight standard
argument-hint: [path or "staged"]
allowed-tools: Read, Grep, Glob, Bash
---

Audit $ARGUMENTS (default: current `git diff`) against the Polaris standard. Delegate to **fsw-code-reviewer** for flight/lib C++; apply the lighter bar to `sim/`, `analysis/`, and tests.

Check and report by severity (🔴 critical / 🟡 warning / 🟢 suggestion), with file:line and a concrete fix for each:

**Always:**
- Quaternions JPL scalar-first via the project library (no ad-hoc quaternion math).
- Time in **TAI**; GNSS GPS-time/ECEF converted before inertial use.
- **SI units**; vectors **frame-tagged**, boundary typed wrappers at interfaces.
- Consumes canonical `EstimatedState`; **no path for `TruthState` to reach the FSW**.
- Every algorithm/model cites a **reference** (`refs.bib`); docstrings declare units + frames.
- Physical parameters come from config, not hard-coded (no defaults exist).
- New algorithm has a test (GMAT-validated where applicable) and a `REQ-###` trace.

**Flight paths only (usually critical):**
- No heap after init; **fixed-size Eigen only**; **no exceptions**; no recursion; bounded loops; return codes checked; finiteness/range checks on outputs.
- Inter-component comms only via typed ports.

End with a one-line verdict: clean, or N critical / M warnings to address before merge. Make no edits.
