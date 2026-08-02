Actuators & Allocation (ACT)
============================

Actuator models and allocation requirements. Source: design doc §7, §8.5, §18.9.
Firm seeds below.

.. req:: RW-or-CMG, not both
   :id: REQ-ACT-001
   :status: reviewed
   :level: L2
   :tags: adcs, actuators, architecture
   :method: Inspection
   :derived_from: REQ-SYS-008
   :allocation: flight/PolarisFsw/ActuatorAllocation

   A vehicle configuration **shall** use either reaction wheels (pyramidal) **or**
   CMGs as its primary momentum actuator, never both; the choice is a modular,
   config-selected actuator.

.. req:: RW allocation law
   :id: REQ-ACT-002
   :status: reviewed
   :level: L2
   :tags: adcs, actuators, allocation
   :method: Test
   :derived_from: REQ-ACTL-003
   :allocation: flight/PolarisFsw/ActuatorAllocation
   :refs: markley2014

   For RW-equipped vehicles the FSW **shall** distribute commanded body torque
   across the wheel pyramid using an L-norm / L-∞ allocation (min-effort /
   min-max torque).

.. req:: Momentum management and desaturation
   :id: REQ-ACT-003
   :status: reviewed
   :level: L2
   :tags: adcs, actuators, momentum
   :method: Test
   :derived_from: REQ-ACTL-003
   :allocation: flight/PolarisFsw/MomentumManager

   The FSW **shall** monitor stored RW momentum and desaturate using magnetorquers
   (and/or thrusters), retaining control authority with margin to wheel saturation.

.. req:: CMG singularity-robust steering
   :id: REQ-ACT-004
   :status: reviewed
   :level: L2
   :tags: adcs, actuators, cmg
   :method: Test
   :derived_from: REQ-ACT-001
   :allocation: flight/PolarisFsw/ActuatorAllocation
   :refs: markley2014

   For CMG-equipped vehicles the FSW **shall** provide a singularity-robust
   steering law (singularity-robust inverse / null-motion) honoring gimbal limits.
