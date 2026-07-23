Simulation & Environment (SIM)
==============================

Source: design doc §5, §6, §7, §2.3, §23.1.1. Fully populated in Phases 1–2; firm
seeds below.

.. req:: 6DOF dynamics with RK89
   :id: REQ-SIM-001
   :status: reviewed
   :level: L2
   :tags: sim, dynamics
   :method: Analysis
   :derived_from: REQ-MIS-002
   :allocation: sim/dynamics
   :refs: montenbruck2000

   The truth sim **shall** integrate coupled translational + rotational rigid-body
   dynamics with an RK89 integrator (configurable tolerance/step), with
   energy/momentum conservation available as validation diagnostics.

.. req:: Environment model suite
   :id: REQ-SIM-002
   :status: reviewed
   :level: L2
   :tags: sim, environment
   :method: Analysis
   :derived_from: REQ-MIS-002
   :allocation: sim/world, lib/environment
   :refs: vallado2013

   The truth sim **shall** model gravity (EGM2008, settable degree/order; optional
   tides), third-body point-mass from SPICE (DE440/DE441), atmospheric drag
   (NRLMSIS 2.1), SRP with conical eclipse, the geomagnetic field (IGRF-14, WMM
   backup), and disturbance torques (gravity-gradient, aero, SRP, residual dipole).

.. req:: Sensor/actuator truth models with occlusion
   :id: REQ-SIM-003
   :status: reviewed
   :level: L2
   :tags: sim, sensors, actuators
   :method: Test
   :derived_from: REQ-MIS-002
   :allocation: sim/sensors, sim/actuators

   The truth sim **shall** provide sensor and actuator truth models with full error
   stacks (bias/drift, scale factor, misalignment, noise, quantization,
   latency, saturation, validity) and a shared line-of-sight occlusion model
   (Earth limb / Sun / Moon) for optical sensors.

.. req:: Deliberate truth/onboard model divergence
   :id: REQ-SIM-004
   :status: reviewed
   :level: L2
   :tags: sim, architecture
   :method: Inspection
   :derived_from: REQ-SYS-005
   :allocation: sim

   Truth models **shall** deliberately differ from the onboard models (fidelity,
   biases, noise, latency) so the sim exercises the estimators and FDIR against
   realistic model error.

.. req:: First-class scriptable fault injection
   :id: REQ-SIM-005
   :status: reviewed
   :level: L2
   :tags: sim, fdir
   :method: Test
   :derived_from: REQ-FDIR-004
   :allocation: sim/sensors, sim/actuators, sim/scenario

   Every sensor/actuator/subsystem model **shall** expose scriptable
   fault-injection hooks (bias jumps, dropouts, occlusions, GPS outage/spoofing,
   stuck/runaway actuators, subsystem limits) so the FDIR suite can drive them per
   scenario.
