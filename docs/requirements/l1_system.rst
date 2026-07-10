L1 — System & Program
=====================

System-level and cross-cutting program requirements, allocated across FSW, sim,
lib, and ground. These are the "Golden Rules" and resolved §18 decisions expressed
as verifiable requirements. Source: design doc §2, §3, §18.

.. req:: TAI onboard master clock
   :id: REQ-SYS-001
   :status: reviewed
   :level: L1
   :tags: time, conventions
   :method: Test
   :derived_from: REQ-MIS-001
   :allocation: lib/time
   :refs: vallado2013

   All onboard logic **shall** run on a TAI master timescale represented as a
   monotonic signed 64-bit integer count of nanoseconds; UTC is derived for ground
   use only.

.. req:: SI units throughout
   :id: REQ-SYS-002
   :status: reviewed
   :level: L1
   :tags: units, conventions
   :method: Inspection
   :derived_from: REQ-MIS-001

   All quantities inside the FSW and ``lib/`` **shall** be expressed in SI base
   units; display-unit conversions occur only at the ground/analysis boundary.

.. req:: JPL scalar-first quaternions
   :id: REQ-SYS-003
   :status: reviewed
   :level: L1
   :tags: attitude, conventions
   :method: Inspection
   :derived_from: REQ-MIS-001
   :refs: markley2014

   All attitude quaternions **shall** use the JPL convention, scalar-first
   ``[q0, q1, q2, q3]`` with canonical ``q0 >= 0``, computed by a single
   quaternion library.

.. req:: Frame-tagged boundary types
   :id: REQ-SYS-004
   :status: reviewed
   :level: L1
   :tags: frames, conventions
   :method: Inspection
   :derived_from: REQ-MIS-001

   Every vector and quaternion crossing a module/port/state boundary **shall**
   carry a compile-time frame tag, such that mixing frames across a boundary is a
   compile error.

.. req:: Single canonical state; truth isolation
   :id: REQ-SYS-005
   :status: reviewed
   :level: L1
   :tags: state, architecture
   :method: Inspection
   :derived_from: REQ-MIS-001

   The GNC chain **shall** use exactly one navigation product (``EstimatedState``
   onboard, ``TruthState`` in sim), and the FSW **shall** be structurally unable
   to consume ``TruthState``.

.. req:: Flight memory and exception model
   :id: REQ-SYS-006
   :status: reviewed
   :level: L1
   :tags: flight, safety
   :method: Inspection
   :derived_from: REQ-MIS-001

   Flight paths (``flight/`` and flight paths of ``lib/``) **shall** perform no
   dynamic memory allocation after initialization, use fixed-size Eigen only, use
   no recursion and only bounded loops, and raise no C++ exceptions; faults are
   surfaced as F´ events.

.. req:: Deterministic reproducibility
   :id: REQ-SYS-007
   :status: reviewed
   :level: L1
   :tags: determinism, sim, mc
   :method: Test
   :derived_from: REQ-MIS-002

   A run **shall** be bit-for-bit reproducible from ``{config, seed}`` at any
   execution speed, including across the two-process SITL boundary; stochastic
   sources draw from explicitly seeded streams.

.. req:: Config as single source of truth
   :id: REQ-SYS-008
   :status: reviewed
   :level: L1
   :tags: config, architecture
   :method: Inspection
   :derived_from: REQ-MIS-001

   Spacecraft/scenario configuration plus the hardware model library **shall** be
   the single source of truth, compiled into F´ params, sim setup, and analysis
   inputs by one config compiler; on-orbit uplinks are a versioned overlay on the
   ground baseline.

.. req:: Reference provenance
   :id: REQ-SYS-009
   :status: reviewed
   :level: L1
   :tags: docs, provenance
   :method: Inspection
   :derived_from: REQ-MIS-005

   Every algorithm, model, filter, and numerical method **shall** cite its source
   (textbook chapter preferred, else paper) in the code header/docstring, keyed to
   ``docs/refs.bib``.

.. req:: GMAT golden verification; no Orekit
   :id: REQ-SYS-010
   :status: reviewed
   :level: L1
   :tags: vv, golden
   :method: Analysis
   :derived_from: REQ-MIS-005
   :refs: gmat2026

   Numerical functions (propagation, time/coordinate conversions, frame
   transforms, eclipse, contacts) **shall** be validated against versioned GMAT
   golden fixtures within documented per-quantity tolerances. Orekit shall not be
   used.

.. req:: Two-process SITL over F´ TCP, sim-time locked
   :id: REQ-SYS-011
   :status: reviewed
   :level: L1
   :tags: sitl, architecture
   :method: Test
   :derived_from: REQ-MIS-002
   :refs: fprime2018

   The sim and FSW **shall** run as two processes coupled over the F´ native
   byte-stream TCP transport, lockstepped on sim time via a per-macro-step
   handshake; the sim owns the master clock.

.. req:: Configurable control cycle
   :id: REQ-SYS-012
   :status: reviewed
   :level: L1
   :tags: cdh, control
   :method: Test
   :derived_from: REQ-MIS-001

   The onboard control cycle **shall** default to 10 Hz and be configurable per
   mission, driven by F´ rate groups (not hard-coded).

.. req:: Comprehensive health telemetry; multi-frame derived views
   :id: REQ-SYS-013
   :status: reviewed
   :level: L1
   :tags: telemetry
   :method: Demonstration
   :derived_from: REQ-MIS-001

   The FSW **shall** publish comprehensive health telemetry and the vehicle state
   in multiple frames/parameterizations (ECI, ECEF, Keplerian, geodetic, LVLH/RIC)
   as derived views of the single ``EstimatedState``.

.. req:: Time persistence across resets
   :id: REQ-SYS-014
   :status: reviewed
   :level: L1
   :tags: cdh, time, persistence
   :method: Test
   :derived_from: REQ-SYS-001

   The onboard clock **shall** persist across resets and be re-acquired from GPS
   time at first fix on cold start.
