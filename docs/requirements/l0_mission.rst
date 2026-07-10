L0 — Mission
============

Top-level mission objectives and constraints. Everything below traces up to one of
these. Source: design doc §1 (Vision & Scope).

.. req:: Onboard GNC flight software
   :id: REQ-MIS-001
   :status: reviewed
   :level: L0
   :tags: mission, fsw
   :method: Demonstration

   Polaris **shall** provide a complete onboard GNC flight software suite for a
   6DOF Earth-orbiting spacecraft — sensing, attitude/orbit estimation, guidance,
   control, actuation, and fault management — demonstrated in closed-loop SITL.

.. req:: High-fidelity verification plant
   :id: REQ-MIS-002
   :status: reviewed
   :level: L0
   :tags: mission, sim
   :method: Demonstration

   Polaris **shall** provide a high-fidelity 6DOF truth/environment simulation
   that serves as the plant the flight software is verified against.

.. req:: Ground analysis tooling reusing flight code
   :id: REQ-MIS-003
   :status: reviewed
   :level: L0
   :tags: mission, analysis
   :method: Demonstration

   Polaris **shall** provide ground analysis tooling (sizing, budgeting,
   scheduling, link analysis) that reuses the same C++ implementations that fly,
   so analysis and flight cannot silently diverge.

.. req:: Safe/degraded operation
   :id: REQ-MIS-004
   :status: reviewed
   :level: L0
   :tags: mission, fdir, adcs
   :method: Test

   Polaris **shall** maintain a power- and thermal-safe, controllable attitude in
   degraded conditions, including a coarse attitude mode when star trackers are
   unavailable.

.. req:: Verification with traceability and margin
   :id: REQ-MIS-005
   :status: reviewed
   :level: L0
   :tags: mission, vv
   :method: Inspection

   Every Polaris capability **shall** be verified through bidirectional
   requirements traceability, with each requirement met at a quantified margin
   against its threshold.
