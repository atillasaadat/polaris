Attitude Determination (ADET)
=============================

Attitude estimation requirements. Source: design doc §8.1, §8.2. Fully populated
in Phase 4; firm seeds below.

.. req:: Fine-mode MEKF
   :id: REQ-ADET-001
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, mekf
   :method: Test
   :derived_from: REQ-MIS-001
   :allocation: flight/components/AttitudeEstimation
   :refs: markley2014

   In fine mode the FSW **shall** estimate attitude and gyro bias with a
   multiplicative EKF using a unit-quaternion reference (JPL scalar-first) and a
   3-parameter error state, fusing star tracker(s), multi-IMU, and multi-sun-sensor
   measurements.

.. req:: Coarse-mode SS+MAG+IMU determination
   :id: REQ-ADET-002
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, safe
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/components/AttitudeEstimation
   :refs: wertz1978

   When star trackers are unavailable, the FSW **shall** determine attitude from
   sun-sensor and magnetometer vector measurements paired with their inertial
   references (sun ephemeris, onboard IGRF-14), propagated by IMU gyros between
   updates.

.. req:: Deterministic single-frame initializers
   :id: REQ-ADET-003
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, init
   :method: Test
   :derived_from: REQ-ADET-001
   :allocation: lib/gnc
   :refs: markley2014, black1964, shuster1981

   The FSW **shall** seed the coarse solution and the MEKF from two vector
   measurements using a deterministic single-frame initializer (TRIAD / QUEST /
   q-method), giving a defined cold-start/acquisition path.

.. req:: Telemetered estimation mode and consistency
   :id: REQ-ADET-004
   :status: reviewed
   :level: L2
   :tags: adcs, telemetry, fdir
   :method: Demonstration
   :derived_from: REQ-SYS-013
   :allocation: flight/components/AttitudeEstimation

   The active estimation mode (fine/coarse) **shall** be telemetered, and
   estimator consistency (NEES/NIS) provided as a diagnostic; fine↔coarse
   transitions are driven by sensor validity and surfaced to FDIR.
