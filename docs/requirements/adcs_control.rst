Attitude Control (ACTL)
=======================

Attitude control law requirements. Source: design doc §8.5. Fully populated in
Phase 5; firm seeds below.

.. req:: B-dot detumble
   :id: REQ-ACTL-001
   :status: reviewed
   :level: L2
   :tags: adcs, control, detumble
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/PolarisFsw/Control
   :refs: markley2014

   The FSW **shall** provide a B-dot detumble control law using magnetorquers to
   reduce body rates from tip-off to within a configured threshold.

.. req:: PID pointing control
   :id: REQ-ACTL-002
   :status: reviewed
   :level: L2
   :tags: adcs, control
   :method: Test
   :derived_from: REQ-MIS-001
   :allocation: flight/PolarisFsw/Control

   The FSW **shall** provide reaction-wheel-based PID attitude control that tracks
   the guidance reference attitude and rate within configured pointing-accuracy and
   settling-time budgets.

.. req:: Actuator-agnostic commanded torque
   :id: REQ-ACTL-003
   :status: reviewed
   :level: L2
   :tags: adcs, control, architecture
   :method: Inspection
   :derived_from: REQ-MIS-001
   :allocation: flight/PolarisFsw/Control

   Guidance/control **shall** emit a commanded body torque, keeping upstream logic
   actuator-agnostic; the allocation layer (RW-pyramid or CMG-steering) is the
   swappable piece (see :doc:`adcs_actuators`).
