---
description: Validate a Polaris numerical function against a GMAT golden fixture
argument-hint: <function or module> [scenario]
---

Validate **$ARGUMENTS** against GMAT golden data. Delegate to the **test-vv** subagent. Steps:

1. Identify the function/module under test and the quantity it produces (propagation, time/coordinate conversion, frame transform, eclipse, contact geometry, …).
2. Check `tests/golden/` for an existing fixture. If none exists, generate one via the GMAT harness in `tools/gmat/` for the given scenario, and store it **versioned** with its GMAT script and the **documented per-quantity tolerance band**.
3. Run the comparison and report: max/RMS error per quantity vs tolerance, **PASS/FAIL with margin**, and the epoch/frames/units of comparison.
4. If it FAILS, do not loosen the tolerance to pass. Report the discrepancy precisely (expected vs actual, where they diverge) and hand off to **gnc-algorithms** or **sim-environment** as a likely product bug.
5. Confirm the test is wired into CI regression and traced to a `REQ-###`.

Remember: **GMAT is the reference; Orekit is not used.** Return the validation result and fixture provenance.
