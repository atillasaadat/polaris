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
   :allocation: flight/PolarisFsw/OrbitEstimator, lib/gnc
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

.. req:: Onboard force-model fidelity and its process-noise budget
   :id: REQ-ODP-005
   :status: reviewed
   :level: L3
   :tags: od, estimation, forcemodel
   :method: Test
   :derived_from: REQ-ODP-001
   :allocation: lib/gnc
   :refs: montenbruck2000, cunningham1970, pavlis2012
   :value_required: >= degree/order 8 geopotential; q_a derived as 3*dr(T)^2/T^3

   The onboard force model **shall** evaluate the geopotential to at least
   degree and order 8 in the Earth-fixed frame, and the filter's process-noise
   acceleration PSD **shall** be derived from the measured divergence between
   that model and a higher-fidelity reference over the coast horizon
   (``q_a = 3*dr(T)^2/T^3``) rather than tuned.

   Rationale: the estimate is receiver-bound while GNSS is available, but the
   dynamics is load-bearing for outage coasting and onboard ephemeris
   prediction, where the previously flown closed-form J2 truncation was the
   entire error budget. The reference used for the measurement must itself be of
   higher fidelity than the onboard model — a characterisation against a
   matched-fidelity reference reports a small number and asserts nothing.

.. req:: GNSS fix-latency correction
   :id: REQ-ODP-006
   :status: reviewed
   :level: L3
   :tags: od, estimation, timing
   :method: Test
   :derived_from: REQ-ODP-001
   :allocation: lib/gnc
   :refs: barshalom2001, kim2025
   :value_required: bounded by max_fix_latency_s; residual O(tau^3)

   The FSW **shall** apply a GNSS fix at the epoch the fix was *measured*, not
   the epoch it was received: a fix tagged behind the filter's current epoch
   **shall** have its reported position and velocity advanced to that epoch on
   the receiver's own reported velocity and the onboard acceleration, with the
   measurement covariance inflated for the advance, and **shall** be refused
   when the latency exceeds a configured bound or when the fix carries no
   velocity.

   Rationale: at LEO orbital speed the delivery latency is metres of along-track
   position per millisecond, an order above the receiver's own accuracy.
   Advancing the measurement is preferred to retrodicting the filter because the
   fix determines the whole state and carries its own velocity, so the linear
   term is measured rather than modelled. The filter's own velocity must not be
   substituted for a missing one: that would fold the filter's error into a
   measurement required to be independent of it.

.. req:: Onboard position served across a receiver outage
   :id: REQ-ODP-007
   :status: reviewed
   :level: L3
   :tags: od, estimation, fdir
   :method: Test
   :derived_from: REQ-ODP-001
   :allocation: flight/PolarisFsw/OrbitEstimator, flight/PolarisFsw/AttitudeEstimator
   :value_required: FINE through max_coast_s after the last accepted fix; DEGRADED through max_degraded_coast_s; dropped past it

   The FSW **shall** publish the onboard orbit solution once per GNC cycle to
   every consumer that needs the vehicle's position — the attitude estimator's
   magnetic and sun references first — with a **quality**: FINE through the
   configured fine coast horizon after the last accepted fix, DEGRADED from
   there through the configured degraded horizon (the solution coasted on the
   force model with the covariance the process noise grows), and dropped past
   it. A consumer **shall** gate on the published position uncertainty against
   its own tolerance rather than on the fine horizon alone, so that a receiver
   outage costs a metre-class consumer its accuracy and a kilometre-class one
   (the magnetic reference: ~1e-3 deg per km) nothing; a consumer **shall**
   treat an absent solution as no position, never reuse the last one. Fixes
   returning inside the degraded horizon **shall** be absorbed by update on the
   grown covariance, not by a re-seed.

   Verified by the SITL rows of ``tests/integration/sitl_od_fault_test.cpp``
   (outages on each side of the fine horizon, a spoof step on each side of it —
   refused throughout on the degraded horizon — a stale receiver clock,
   ``OD_RESET`` mid-run, a second receiver carrying a primary outage, one orbit
   period with no fault) and ``sitl_od_burn_test.cpp`` (a 1200 s outage coasted
   DEGRADED to 2.2 m with the magnetic reference kept and no re-seed).

   Rationale: until this seam existed the attitude estimator read the receiver
   directly and a GNSS outage cost the magnetic pair on the first missed fix.
   Serving position from the filter moves that dependency onto a horizon sized
   from the receiver's own outage modes (§8.3) and puts the plausibility gate on
   the fix, in one place, ahead of every consumer.

.. req:: Non-gravitational acceleration input and ground seed
   :id: REQ-ODP-008
   :status: reviewed
   :level: L3
   :tags: od, estimation, maneuver
   :method: Test
   :derived_from: REQ-ODP-001, REQ-ODP-007
   :allocation: lib/gnc, flight/PolarisFsw/OrbitEstimator, flight/PolarisFsw/BurnExecutor
   :value_required: a burn inside an outage coasted to < 100 m at outage end (measured 26 m told vs 1374 m blind)
   :refs: ceresoli2025

   The onboard orbit filter **shall** accept a known non-gravitational
   acceleration — the commanded thrust of a finite burn (§17), in the inertial
   frame with a 1-sigma magnitude — and **shall** propagate with it over the
   step and inflate its process noise by that sigma over the same step, so a
   burn is neither refused fix by fix nor coasted through blind; an invalid or
   stale acceleration record **shall** mean "no thrust known", never the last
   value. The filter **shall** accept a ground-supplied state seed (epoch,
   position, velocity, sigmas) for a long outage or a failed receiver, refused
   when the epoch is older than the degraded horizon or in the future beyond
   the fix-latency bound.

   Rationale: Ceresoli et al. (2025) measured a burn during a GNSS outage as
   5 km of error with the thrust fed from the IMU against 9 km propagating
   the last fix, on a J2 model; on the 8x8 model here, the same burn inside a
   900 s outage is 26 m told against 1374 m blind, and a 60 s burn while
   tracking is absorbed at the receiver's noise (0.13 m). The accelerometer path
   is deliberately not the source yet: an accelerometer bias of hundreds of
   micro-g integrated over a coast is worse than no input, so it waits on a bias
   state and a burn-active gate.

   Verified by ``tests/unit/orbit_od_test.cpp``
   (``NonGravitationalAccelerationIsPropagatedAndBudgeted``, the seed API), the
   MC scenarios ``burn_tracked`` / ``burn_outage_fed`` / ``burn_outage_blind``
   (``tests/mc``), the ``OrbitEstimator`` component tests
   (``NonGravAccelIsAppliedOnlyWhenFreshAndValid``, ``GroundSeedAcceptedAndRefused``)
   and the SITL rows of ``tests/integration/sitl_od_burn_test.cpp``.
