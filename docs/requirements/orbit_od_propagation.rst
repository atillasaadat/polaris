Orbit Determination & Propagation (ODP)
=======================================

Source: design doc §8.3, §11. Fully populated in Phase 6; firm seeds below.

.. req:: Onboard MEKF orbit determination with self-covariance
   :id: REQ-ODP-001
   :status: reviewed
   :level: L2
   :tags: od, estimation
   :method: Test
   :derived_from: REQ-MIS-001
   :allocation: flight/PolarisFsw/OrbitEstimation
   :refs: montenbruck2000

   The FSW **shall** estimate its own orbit with an onboard MEKF from GNSS-sim
   measurements and the onboard force model, propagating self-covariance; GNSS
   inputs are converted GPS→TAI and ECEF→ECI before the filter runs.

.. req:: Multi-object propagation with covariance
   :id: REQ-ODP-002
   :status: reviewed
   :level: L2
   :tags: od, multiobject
   :method: Test
   :derived_from: REQ-ODP-001
   :allocation: flight/PolarisFsw/OrbitEstimation
   :value_required: N ~ 5 secondaries

   The FSW **shall** propagate up to N (~5) secondary objects from an uploaded
   state-vector + covariance or TLE, with covariance propagation, using the onboard
   force model (or SGP4 for TLE-sourced objects).

.. req:: SGP4 propagation and CCSDS OEM product
   :id: REQ-ODP-003
   :status: reviewed
   :level: L2
   :tags: od, interop
   :method: Analysis
   :derived_from: REQ-SYS-010
   :allocation: lib/gnc
   :refs: vallado2013

   The suite **shall** provide SGP4 TLE propagation and generate CCSDS OEM
   ephemeris as a data product, validated against GMAT golden fixtures.

.. req:: Ground batch least-squares OD
   :id: REQ-ODP-004
   :status: reviewed
   :level: L2
   :tags: od, ground
   :method: Analysis
   :derived_from: REQ-MIS-003
   :allocation: analysis, lib/gnc
   :refs: vallado2013

   The ground/analysis tooling **shall** provide a batch least-squares orbit
   estimator for orbit fit and validation.
