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

.. req:: Navigation-filter usability practices — retune, covariance re-initialisation, selective processing, backup ephemeris
   :id: REQ-ODP-009
   :status: reviewed
   :level: L3
   :tags: od, estimation, operability, fdir
   :method: Test
   :derived_from: REQ-ODP-001, REQ-ODP-007
   :allocation: lib/gnc, flight/PolarisFsw/OrbitEstimator
   :refs: carpenter2018, dennehy2020

   The onboard orbit filter **shall** implement the NESC navigation-filter
   usability practices of NASA/TP-2018-219822 Ch. 9 and NESC Technical Bulletin
   20-03 items (d)–(g):

   * **(g) tuning without loss of navigation data** (TP §9.3): a parameter
     upload **shall** re-tune the running filter in place — state, covariance,
     epoch, age and counters kept — and a set that fails validation **shall**
     leave the last valid set in force rather than make a running filter inert;
   * **(f) covariance re-initialisation without altering the state** (TP §9.2):
     ``OD_REINIT_COV(posSigma, velSigma)`` **shall** set the covariance to the
     commanded isotropic values and leave position, velocity and age untouched;
   * **(d) selective processing per measurement type** (TP §9.1): a three-way
     ``ACCEPT`` / ``INHIBIT`` / ``FORCE`` policy **shall** be uplinkable per
     measurement type (position, velocity); ``INHIBIT`` **shall** withhold the
     type regardless of the residual-edit (NIS) test, ``FORCE`` **shall** apply it
     regardless of that test, and a forced update **shall** be counted apart from
     both acceptances and rejections. ``FORCE`` **shall not** override a numeric
     fault (a negative or non-finite NIS);
   * **(e) a backup ephemeris** (TP §9.2): a copy of the solution, unaltered by
     measurement updates since it was seeded and re-seeded from a FINE solution
     every ``BackupPeriodS``, **shall** be propagated alongside the filter;
     ``OD_RESTART_FROM_BACKUP`` **shall** restore the solution from it without an
     uplinked state vector, and the separation between the two **shall** be
     telemetered as an independent divergence comparator.

   The covariance **shall** be checked for positive semi-definiteness every cycle
   (TP Ch. 7 — the check a UDU filter gets for free from ``D``) and an
   indefinite covariance reported by event.

   Rationale: these are the operability rules NASA's navigators codified from
   Gemini/Apollo through Shuttle and Orion, "without a failure ever attributed
   to an EKF" [dennehy2020]. Before Push 71 a parameter upload rebuilt the filter
   and dropped a converged solution to change a gate, the only recovery from an
   over-confident filter was ``OD_RESET`` (which needed a new fix) or
   ``OD_SEED_STATE`` (which needed an uplink), and there was no way for the
   ground to withhold or force a measurement type. Underweighting (TP Ch. 4) is
   deliberately **not** adopted: the GNSS position and velocity measurement
   models are linear (``H = [I 0]``, ``[0 I]``), so the second-order term the
   technique compensates is identically zero here. The UDU factorisation (TP
   Ch. 7) is likewise not adopted for a 6-state double-precision filter with a
   Joseph update and explicit symmetrisation; only its definiteness check is.

   Verified by ``tests/unit/orbit_od_test.cpp``
   (``OrbitOdUsability.RetuneKeepsTheSolutionAndRefusesABadConfig``,
   ``CovarianceReinitialisationKeepsTheState``,
   ``MeasurementPolicyInhibitsAndForces``) and the ``OrbitEstimator`` component
   tests ``TuningUploadKeepsTheSolution``,
   ``CovarianceReinitAndMeasurementPolicy`` and ``BackupEphemerisRestart``.

.. req:: Time-scale exposure of the orbit filter — a misapplied leap second is refused, not absorbed
   :id: REQ-ODP-010
   :status: reviewed
   :level: L3
   :tags: od, time, fdir
   :method: Test
   :derived_from: REQ-ODP-001, REQ-CONV-001
   :allocation: lib/gnc, lib/frames
   :refs: carpenter2018

   The orbit filter **shall** run on a continuous time scale (TAI) internally
   (NASA/TP-2018-219822 §6.3), and the one place a discontinuous scale reaches
   it — the leap-second table (ΔAT) that the ECEF→ECI reduction of a GNSS fix
   passes through on the way to UT1 — **shall** be a refusal, not an absorption:
   a tracking filter offered a fix converted through a leap-second table stale
   by one leap (a 15 arcsec Earth-rotation error, ~500 m at LEO) **shall** refuse
   it on the NIS gate and leave the solution untouched.

   Recorded limit: a **cold** filter has no prior and therefore no gate, so it
   seeds on such a fix and flies a self-consistent solution rotated by that
   angle. Consumers that work in ECEF (the magnetic reference, ground-station
   geometry) round-trip through the same table and are unaffected; the exposure
   is confined to inertial consumers, and to a table that is stale at cold
   start — which the §11.3 table-validity monitoring is what should catch.

   Verified by ``tests/unit/orbit_od_test.cpp``
   (``OrbitOdTimeScales.AStaleLeapSecondTableIsRefusedByATrackingFilterAndOnlyRotatesASeed``),
   which measures both the refusal (innovation 300–700 m, solution within 5 m of
   truth) and the seed-path rotation.

.. req:: Process-noise structure, DMC acceleration states and covariance metrics
   :id: REQ-ODP-011
   :status: reviewed
   :level: L3
   :tags: od, estimation, tuning
   :method: Test
   :derived_from: REQ-ODP-001, REQ-ODP-005
   :allocation: lib/gnc, flight/PolarisFsw/OrbitEstimator, tests/mc, analysis/od
   :refs: carpenter2018

   The onboard orbit filter **shall** carry the process-noise structure and the
   covariance metrics of NASA/TP-2018-219822 Ch. 2:

   * **State-noise compensation in orbit-fixed axes** (TP §2.2.3.1): a
     per-axis RTN acceleration PSD ``(q_R, q_T, q_N)``, rotated into the
     inertial frame at each sub-step and added to the isotropic PSD, so the
     along-track intensity is a separate tuning knob for secular along-track
     growth (TP §2.2.4.2); TP Eq. 2.88 **shall** be available as the starting
     point for it (``alongTrackPsdFromOneOrbitError``, and the same in
     ``analysis/od``);
   * **Dynamic model compensation** (TP §2.2.3.3): three exponentially
     correlated (first-order Gauss-Markov) acceleration states in RTN with a
     configurable correlation time and PSD, added to the force model, their
     transition and discrete process noise per TP Eqs. 2.54–2.55, disabled at
     ``τ = 0``; the position/velocity marginal **shall** remain the product
     every consumer reads;
   * **Covariance metrics** (TP §2.1): the semi-major-axis 1σ (Eq. 2.23) and
     the flight-path-angle 1σ (Eq. 2.26) **shall** be computed from the
     solution's covariance and telemetered, and the Monte Carlo campaign
     **shall** record the SMA error and SMA sigma per sample so the covariance
     is judged on the metric that predicts (§2.1.4).

   Rationale: SMA error is period error is secular along-track drift, and the
   TP names it the OD figure of merit; the RTN form is how the along-track
   growth is tuned; the DMC states are the TP's answer to the systematic
   truncation error this filter otherwise absorbs as white noise (REQ-ODP-005).

   **What is flown, and why (measured, 2026-08-18).** The reference vehicle
   ships with the RTN intensities at zero and the DMC states off — the Push 65
   isotropic ``q_a`` unchanged. On the ``outage_horizon`` scenario (3 runs,
   20 000 s) a DMC layer at τ = 600 s / σ_a = 3e-5 m/s² **worsened** the coast
   error (outage worst 2.35 → 3.6 m; τ = 100 s: 3.8 m) with the campaign NEES
   still consistent — the 8×8 truncation on this truth is not well described
   by a 600 s Gauss-Markov acceleration at 1 Hz fixes, and the states carry
   noise into the coast; reducing ``q_a`` alone by 10× drove the NEES to 24
   against a 10.5 upper bound (optimistic, as Push 65 measured). An RTN
   restructuring (``q_iso`` halved, ``q_T`` = the old ``q_a``) **improved** the
   coast (outage worst 2.35 → 2.05 m, 95th 2.11 → 1.87 m) at NEES 6.75 inside
   [2.74, 10.5] — a candidate the 30-run campaign gate, not a 3-run sample,
   has to sign off. The MC harness takes ``--q-rtn``, ``--qa-scale``,
   ``--dmc-tau-s``, ``--dmc-psd`` so that study runs on the same shards.

   Verified by ``tests/unit/orbit_od_dmc_test.cpp``
   (``NoiseKernelMatchesHighOrderQuadratureAndItsSmallStepLimit``,
   ``RtnBasisIsOrthonormalAndOriented``,
   ``RtnProcessNoiseGrowsTheVelocityAlongTheAxisItNames``,
   ``UnmodelledAccelerationIsEstimatedAndClosesTheCoast`` — a 2e-5 m/s²
   along-track acceleration estimated to within 50 % and a 300 s coast closed
   to under half the blind error,
   ``SmaAndFlightPathAngleSigmasMatchTheClosedFormsOnACircularOrbit``), the
   ``OrbitEstimator`` component test ``CovarianceMetricsAndDmcParameters``, and
   ``tests/analysis/test_od_statistics.py`` (SMA summaries, Eq. 2.88).

.. req:: Bounded re-acquisition after an unmodelled event
   :id: REQ-ODP-012
   :status: reviewed
   :level: L3
   :tags: od, estimation, fdir
   :method: Test
   :derived_from: REQ-ODP-001, REQ-ODP-009
   :allocation: lib/gnc, tests/mc, analysis/od
   :refs: carpenter2018

   After an event the filter was given no knowledge of — a burn it was not fed,
   a spoof it rode, a bad fix it accepted — the estimate is offset by more than
   its covariance admits and the innovation gate refuses honest fixes until the
   solution is let go and the next fix re-seeds whole. That recovery **shall**
   be bounded by the configured degraded coast horizon: the longest unbroken
   stretch of delivered, un-faulted fixes the gate refuses **shall not** exceed
   ``max_degraded_coast_s`` plus one GNC cycle.

   The bound is the filter's own policy, not a figure chosen for the campaign.
   Nothing re-seeds sooner by construction: an "N rejections → reseed" rule is
   refused deliberately so a persistent spoof cannot win the seed early
   (design doc §9.2), which makes the horizon both the guarantee and the price.

   The campaign measures the stretch rather than a rate. A rate averages a
   lockout over the whole arc, so a filter that refuses every honest fix for
   half an hour and one that scatters the same refusals across a day read
   alike, and only the first is a filter that cannot get back. The clean-fix
   rejection rate of REQ-ODP-001 is therefore scoped to the samples where the
   filter still reports its solution *fine*: a fix refused during a degraded
   coast is refused correctly — the measurement is good and the state is not —
   and counting it as a gate false alarm measured this requirement's latency
   under that requirement's name.

   Verified by the campaign criterion *Clean-fix lockout after an unmodelled
   event* (``analysis/od/report.py``), which is reported per scenario and reads
   its bound from the driver's own meta record, and by
   ``tests/analysis/test_od_statistics.py``
   (``test_the_clean_fix_lockout_measures_the_longest_unbroken_refusal``,
   ``test_a_refusal_the_fault_earned_is_not_a_lockout``,
   ``test_a_fix_refused_while_the_solution_is_degraded_is_not_a_false_alarm``).

.. req:: Onboard drag scale-factor estimation
   :id: REQ-ODP-013
   :status: reviewed
   :level: L3
   :tags: od, estimation, disturbances
   :method: Test
   :derived_from: REQ-ODP-001, REQ-ODP-005
   :allocation: lib/gnc, flight/PolarisFsw/OrbitEstimator, tests/unit
   :refs: carpenter2018, tapley2004, vallado2013

   The onboard force model flies a **static** exponential atmosphere with no
   solar or geomagnetic activity in it, so its density is wrong by a factor of
   order one against the real one. The orbit filter **shall** be able to
   estimate that error as a dimensionless **drag scale factor** — a multiplier
   on the drag term, carried as a first-order Gauss-Markov state about its
   nominal 1 (design doc §8.5 tier 3, orbit half; TP §2.2.3.4; Tapley, Schutz
   & Born §4.16 for the augmented-state form).

   Three properties are required of it, and they matter more than the estimate:

   1. **It publishes its own uncertainty.** Drag is observable only through the
      secular along-track signature it leaves over an arc, so on a short arc, or
      on a vehicle whose drag is small against the rest of the force-model
      error, the estimate *is* its prior. ``dragScaleSigma`` **shall** say so
      rather than the filter reporting a fit it does not have.
   2. **An out-of-band estimate is refused, not clamped.** An exponential
      atmosphere is not wrong by the factor a wild estimate claims, so such an
      estimate was driven by something that is not drag; clamping would fly a
      magnitude the policy chose rather than one the data supported. The
      refusal **shall** hold the last accepted scale and **shall not** disturb
      the position/velocity solution the fix also carried.
   3. **Disabled means absent.** With the process-noise PSD at zero — the flown
      value — the trajectory **shall** be bit-for-bit that of the filter
      without the state.

   **As-delivered condition: the state ships disabled on the reference
   vehicle, and the reason is measured.** The isotropic ``q_a`` is sized from
   the 8x8 geopotential truncation: 1.28 m over the 300 s coast horizon, an
   equivalent constant acceleration of 2.8e-5 m/s². The whole drag-scale signal
   is ``(s-1)·a_drag``, and at 400 km ``a_drag`` is 1.2e-6 m/s², so even a 60 %
   density error is 7.2e-7 m/s² — **39x under the budget the filter already
   carries**. Measured against a truth atmosphere 60 % denser, the estimate
   after six orbits at the flown tuning is **1.023 of a true 1.6** with σ still
   0.33 of its 0.5 prior; at ``q_a``/1000 the same filter reaches 1.516 with σ
   0.158 in one orbit. Arc length is not the lever and ``q_a`` cannot be reduced
   — Push 74 measured every reduced-``q_a`` variant failing the NEES
   consistency gate, precisely because ``q_a`` covers that truncation. What
   unblocks the scale factor here is a **higher-degree onboard geopotential**,
   not a longer pass, and this requirement is written to be satisfiable on the
   vehicle that has one.

   Verified by ``tests/unit/orbit_od_drag_scale_test.cpp`` (the exact Jacobian
   column against a differenced one and against the term's linearity; recovery
   of a known density bias; the flown tuning's measured non-resolution as an
   asserted upper bound; the short-arc sigma; refused-not-clamped; re-seeding
   on an enable-by-uplink; and the configuration refusals) and by
   ``flight/PolarisFsw/OrbitEstimator``'s
   ``DragScaleParametersAndTelemetry``, which covers the parameter path and the
   ``DragScale``/``DragScaleSigma``/``DragScaleRefused`` channels.

.. req:: Correlated GNSS error and the measurement-covariance repair
   :id: REQ-ODP-014
   :status: reviewed
   :level: L3
   :tags: od, estimation, gnss, sensors
   :method: Test
   :derived_from: REQ-ODP-001
   :allocation: sim/sensors, lib/gnc, flight/PolarisFsw/OrbitEstimator, tools/configc

   Real single-point GNSS position error is **not white**: residual ionosphere,
   broadcast ephemeris and satellite clock are common to the satellites in view
   and decorrelate over minutes, not per fix. The receiver model **shall**
   represent that as a first-order Gauss-Markov component of the position error,
   and the orbit filter **shall** carry a measurement-covariance inflation that
   keeps it consistent against one.

   Three properties are required:

   1. **The split preserves the datasheet.** The correlated part is expressed as
      a fraction of the datasheet variance, so the receiver's total accuracy is
      unchanged and only its spectrum differs. Adding a correlated term *on top*
      of the datasheet would model a worse receiver than the one described, and
      would confound "the error is bigger" with "the error is correlated" — only
      the second is what the filter is unprepared for.
   2. **The receiver reports the total and nothing about the colour.** A formal
      solution covariance is derived from range residuals and geometry; it can
      say how big the error is but not how much of it survives to the next fix.
      A fix from a correlated receiver **shall** be indistinguishable, on its
      reported sigmas alone, from one from a white receiver of the same
      datasheet.
   3. **Disabled means absent.** With the correlated fraction at zero the
      receiver's white draws **shall** be bit-identical to the pre-Push-77
      model, and with the inflation at zero the filter **shall** be unchanged.

   **The repair is a tuning, not a formula, and it is sized for persistence
   rather than for magnitude.** A Kalman filter assumes white measurement noise,
   so over the fixes that share one realisation of the correlated error it drives
   its covariance down as though averaging independent samples while the error
   does not average away at all. Inflating ``R`` by the correlated variance
   itself — the textbook remedy for unmodelled measurement error — closes only
   half the gap. Measured on the reference vehicle's 12-run nominal campaign
   against the gate's own chi-square interval [4.202, 8.113]:

   .. list-table::
      :header-rows: 1

      * - Inflation
        - Campaign NEES
        - Verdict
      * - none
        - 23.597
        - **FAIL** (-191 % margin)
      * - 3x the receiver's correlated sigma
        - 8.067
        - PASS by 0.6 % — not a margin
      * - **4x (flown)**
        - **6.331**
        - PASS, 22 % / 51 % margin

   Position RMS pays 1.317 -> 1.400 m for it, which is the honest cost of a
   covariance that is no longer a fiction. The multiple is **not** derivable:
   sweeping fix cadence at fixed correlation time gives ``k ~ 0.58*sqrt(tau/dt)``,
   but sweeping the correlation time at fixed cadence shows ``k`` saturating near
   3.2 once ``tau`` exceeds the timescale on which ``q_a`` reopens the covariance
   anyway. It is therefore **tuned per vehicle against NEES**, enforced as a
   lower bound by ``configc`` (the inflation may not fall below the receiver's
   own correlated sigma), and what removes the need for it altogether is the raw
   pseudorange path (§8.3, still owed), where the common-mode terms have their
   own signature across the satellites in view.

   Verified by ``tests/unit/sim_sensors_gnss_test.cpp`` (the datasheet-preserving
   split, the stationary variance and its cadence invariance, the measured
   autocorrelation against ``exp(-dt/tau)``, the reported-sigma
   indistinguishability, and the disabled bit-identity), by
   ``tests/unit/orbit_od_correlated_gnss_test.cpp`` (the degradation, the
   insufficiency of a 1x inflation, and the restored consistency at the tuned
   value, bounded on both sides), by ``tests/analysis/`` for the reading of the
   campaign, and by the campaign itself.
