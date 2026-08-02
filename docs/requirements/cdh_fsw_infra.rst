C&DH / FSW Infrastructure (CDH)
===============================

Source: design doc §4, §11.3, §23.5, §23.6. Firm seeds below.

.. req:: Deterministic rate-group execution within timing budget
   :id: REQ-CDH-001
   :status: reviewed
   :level: L2
   :tags: cdh, realtime
   :method: Test
   :derived_from: REQ-SYS-012
   :allocation: flight/PolarisFsw/Top
   :value_required: 10 Hz frame closes within WCET budget

   The FSW **shall** execute on F´ rate groups with deterministic, bounded timing;
   the 10 Hz control frame **shall** close within its documented per-rate-group WCET
   timing budget.

.. req:: Onboard ephemeris and EOP; no SPICE onboard
   :id: REQ-CDH-002
   :status: reviewed
   :level: L2
   :tags: cdh, ephemeris, frames
   :method: Test
   :derived_from: REQ-CONV-002
   :allocation: lib/ephemeris

   The FSW **shall** obtain Sun/Moon/planet positions onboard from uploaded
   Chebyshev coefficient sets and use an uploaded EOP + leap-second table; SPICE is
   never used onboard (ground/sim only).

.. req:: Persistent state and restart behavior
   :id: REQ-CDH-003
   :status: reviewed
   :level: L2
   :tags: cdh, persistence
   :method: Test
   :derived_from: REQ-SYS-014
   :allocation: flight/PolarisFsw/Persistence

   The FSW **shall** persist time/epoch, OD state + covariance, ephemeris/EOP
   tables, mode, calibration, and the secondary-object catalog across resets, with
   defined cold/warm-start behavior and graceful resume.

.. req:: Ports-only inter-component communication
   :id: REQ-CDH-004
   :status: reviewed
   :level: L2
   :tags: cdh, architecture
   :method: Inspection
   :derived_from: REQ-MIS-001
   :allocation: flight/PolarisFsw/Top

   FSW components **shall** communicate only through typed F´ ports — no
   back-channels, globals, or shared mutable state outside ports.
