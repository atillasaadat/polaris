Glossary
========

Acronyms and terms used across Polaris (design doc §23.7). Units and frames for
physical quantities follow the conventions in :doc:`requirements/conventions_frames_time`.

.. glossary::
   :sorted:

   GNC
      Guidance, Navigation & Control.

   FSW
      Flight Software — the F´/C++ deliverable that "flies" (``flight/``).

   SITL
      Software-in-the-Loop — the truth sim and FSW run as two processes in a
      sim-time-locked closed loop.

   PIL / HIL
      Processor- / Hardware-in-the-Loop — later-phase test configurations.

   ADCS
      Attitude Determination & Control Subsystem.

   OD
      Orbit Determination.

   MEKF
      Multiplicative Extended Kalman Filter — the attitude (and orbit) estimator.

   TRIAD / QUEST
      Deterministic single-frame attitude initializers from two vector
      measurements.

   RW
      Reaction Wheel (pyramidal configuration in Polaris).

   CMG
      Control Moment Gyroscope. A vehicle uses **either** RWs **or** CMGs.

   MTQ
      Magnetorquer (magnetic torque rod) — used for B-dot and RW desaturation.

   IMU
      Inertial Measurement Unit (gyro + accelerometer); emits delta-angle /
      delta-velocity.

   SS / MAG / ST
      Sun Sensor / Magnetometer / Star Tracker.

   GNSS
      Global Navigation Satellite System receiver (simulated); reports GPS time
      and ECEF.

   TAI
      International Atomic Time — the onboard master timescale (monotonic, int64 ns).

   GPS time
      GNSS timescale; ``TAI = GPS + 19 s``.

   UTC
      Coordinated Universal Time — derived from TAI for ground use only.

   TT / TDB
      Terrestrial Time / Barycentric Dynamical Time — used for ephemeris.

   ECI
      Earth-Centered Inertial frame; **J2000** is the attitude reference.

   ICRF / J2000
      International Celestial Reference Frame / J2000 mean equator-equinox.

   ECEF / ITRF
      Earth-Centered Earth-Fixed / International Terrestrial Reference Frame.

   LVLH
      Local-Vertical/Local-Horizontal frame (nadir/orbit-relative pointing).

   RIC / RTN
      Radial / In-track / Cross-track (a.k.a. Radial/Transverse/Normal, Hill
      frame). Same triad; RIC is canonical.

   Body
      Spacecraft structural frame (B).

   WGS84
      World Geodetic System 1984 — the canonical Earth reference ellipsoid.

   EOP
      Earth Orientation Parameters (UT1-UTC, polar motion); onboard via an
      uploaded table.

   IGRF
      International Geomagnetic Reference Field (IGRF-14) — onboard modeled field.

   WMM
      World Magnetic Model — backup geomagnetic model.

   EGM2008
      Earth Gravitational Model 2008 — truth spherical-harmonic gravity field.

   NRLMSIS
      Naval Research Laboratory Mass Spectrometer and Incoherent Scatter model
      (2.1) — atmospheric density for drag.

   SRP
      Solar Radiation Pressure.

   SGP4
      Simplified General Perturbations model 4 — TLE propagation.

   TLE
      Two-Line Element set.

   OEM / OMM
      Orbit Ephemeris Message / Orbit Mean-elements Message (CCSDS).

   CCSDS
      Consultative Committee for Space Data Systems.

   FDIR
      Fault Detection, Isolation & Recovery.

   FPP
      F´ Prime Prime — the F´ modeling language for components/ports/topology.

   GDS
      F´ Ground Data System.

   EVR
      Event Record — an F´ event used to surface faults (no exceptions in flight).

   WCET
      Worst-Case Execution Time.

   RK89
      Runge-Kutta 8(9) integrator used by the truth sim.

   DCM
      Direction Cosine Matrix.

   RPOD
      Rendezvous, Proximity Operations & Docking (architecture stubs only).

   SoC
      State of Charge (battery).

   EIRP / RSSI
      Effective Isotropic Radiated Power / Received Signal Strength Indicator.

   GMAT
      General Mission Analysis Tool (NASA GSFC) — golden-data generation.

   MC
      Monte Carlo.

   NEES / NIS
      Normalized Estimation Error Squared / Normalized Innovation Squared —
      estimator-consistency metrics.

   REQ
      A requirement, ``REQ-<SUBSYS>-NNN`` (see :doc:`requirements/index`).

   RVTM
      Requirements Verification & Traceability Matrix.

   T/A/I/D
      The four verification methods: Test, Analysis, Inspection, Demonstration.

   SI
      Système International (units) — used everywhere inside FSW and ``lib/``.

   F´
      F Prime — NASA/JPL open-source flight-software framework.
