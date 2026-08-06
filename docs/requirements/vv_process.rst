Verification Process (VV)
=========================

Requirements on *how* Polaris is verified. Source: design doc §13, §22.2, §23.1.
Firm seeds below.

.. req:: Test pyramid
   :id: REQ-VV-001
   :status: reviewed
   :level: L2
   :tags: vv, ci
   :method: Inspection
   :derived_from: REQ-MIS-005
   :allocation: tests

   Verification **shall** follow a test pyramid: unit (math/library) → component
   (F´) → integration (closed-loop SITL) → Monte Carlo → regression
   (golden-file comparison).

.. req:: GMAT golden regression
   :id: REQ-VV-002
   :status: reviewed
   :level: L2
   :tags: vv, golden
   :method: Analysis
   :derived_from: REQ-SYS-010
   :allocation: tests/golden, tools/gmat
   :refs: gmat2026

   Numerical functions **shall** be compared against versioned GMAT golden fixtures
   within documented per-quantity tolerance bands, regenerable by scripted GMAT
   runs.

.. req:: Bidirectional traceability gate
   :id: REQ-VV-003
   :status: reviewed
   :level: L2
   :tags: vv, traceability, ci
   :method: Inspection
   :derived_from: REQ-MIS-005
   :allocation: docs/requirements

   Every capability **shall** trace to >= 1 requirement and every requirement to
   >= 1 verifying test, enforced in CI: a baselined requirement without a passing
   verifier (or below its required margin) fails the docs/traceability build.

.. req:: Monte Carlo pass/fail with margin
   :id: REQ-VV-004
   :status: reviewed
   :level: L2
   :tags: vv, mc
   :method: Analysis
   :derived_from: REQ-MIS-005
   :allocation: mc

   Monte Carlo campaigns **shall** report margin against requirement thresholds
   (percentile-vs-threshold), not just pass/fail, and aggregate estimator
   consistency (NEES/NIS) across runs.

.. req:: FDIR fault-injection suite
   :id: REQ-VV-005
   :status: reviewed
   :level: L2
   :tags: vv, fdir
   :method: Test
   :derived_from: REQ-FDIR-004
   :allocation: tests/integration

   A dedicated closed-loop SITL suite **shall** inject faults and assert FDIR
   detects, isolates, and responds correctly (right event, right
   mode/reconfiguration, recovery) within bounded time-to-detect/respond.

.. req:: FreeFlyer independent cross-validation
   :id: REQ-VV-006
   :status: reviewed
   :level: L2
   :tags: vv, golden, freeflyer
   :method: Analysis
   :derived_from: REQ-SYS-010
   :allocation: tools/freeflyer, tests/freeflyer
   :refs: gmat2026

   Propagation and attitude-kinematics behaviour **shall** be cross-validated
   against a third, fully independent astrodynamics implementation (a.i.
   solutions FreeFlyer) by replaying the GMAT golden propagation cases through
   the FreeFlyer Runtime API and comparing sampled states within documented
   per-case tolerance bands. The comparison is three-way by construction: the
   C++ stack verifies against the golden fixtures in ``polaris_golden_tests``,
   and FreeFlyer verifies against the same fixtures here, so a disagreement
   isolates the odd implementation out. The suite runs wherever a licensed
   FreeFlyer installation is discovered and skips — visibly, never silently
   passing — where none is.

   Measured on FreeFlyer 7.10.1 (2026-08-05): worst position disagreement
   across all eight cases is **0.27 m** (geo, SRP-flux convention difference)
   with every LEO case under 0.25 m, and the attitude spinner agrees to
   **< 1e-6 deg**. Verified by ``tests/freeflyer/test_propagation_vv.py``.
