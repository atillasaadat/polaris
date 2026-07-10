Fault Management (FDIR)
=======================

Source: design doc §9, §23.1.1. Fully populated in Phase 10; firm seeds below.

.. req:: Tiered FDIR with safing escalation
   :id: REQ-FDIR-001
   :status: reviewed
   :level: L2
   :tags: fdir, safety
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/components/FDIR

   The FSW **shall** implement monitors → isolation → response → safing escalation;
   faults are F´ events with severity, and every alert maps to an autonomous action
   or an operator alarm with a documented (possibly null) action. FDIR state and
   triggers are telemetered.

.. req:: Validity-flag gating
   :id: REQ-FDIR-002
   :status: reviewed
   :level: L2
   :tags: fdir, sensors
   :method: Test
   :derived_from: REQ-SYS-005
   :allocation: flight/components/SensorProcessing

   Each measurement **shall** carry a validity flag (range, rate-of-change,
   staleness/timeout, cross-sensor consistency, solution-quality), and downstream
   consumers **shall** exclude invalid measurements — never silently use them.

.. req:: Estimator and GNSS fault response
   :id: REQ-FDIR-003
   :status: reviewed
   :level: L2
   :tags: fdir, gnss, estimation
   :method: Test
   :derived_from: REQ-FDIR-001
   :allocation: flight/components/FDIR

   The FSW **shall** trigger fine→coarse attitude fallback on star-tracker
   loss/occlusion, and detect GNSS outage (coasting on propagation) and
   spoofing/meaconing (innovation/consistency + reasonableness bounds, with
   measurement rejection).

.. req:: Fault-injection verification
   :id: REQ-FDIR-004
   :status: reviewed
   :level: L2
   :tags: fdir, vv
   :method: Test
   :derived_from: REQ-MIS-005
   :allocation: tests/integration

   Every FDIR monitor/response **shall** be exercised by the fault-injection
   integration suite, asserting correct detection (no false positives on nominal
   runs) and response within a required detection latency.
