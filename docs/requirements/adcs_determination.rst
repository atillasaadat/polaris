Attitude Determination (ADET)
=============================

Attitude estimation requirements. Source: design doc §8.1, §8.2. Fully populated
in Phase 4; firm seeds below.

.. _adet-knowledge-metric:

The attitude-knowledge metric
-----------------------------

The knowledge-accuracy requirements (REQ-ADET-005 … REQ-ADET-007) are all stated
on the **error norm**: the total eigenaxis rotation angle between the estimated
and the true attitude,

.. math::

   \theta_\mathrm{err} = 2 \arccos \left| q_{\mathrm{err},0} \right| ,
   \qquad q_\mathrm{err} = \hat{q} \otimes q_\mathrm{true}^{-1} ,

quoted at the **3σ (99.73rd-percentile)** point of a Monte Carlo campaign over
the sensor error budget. The norm rather than a per-axis split, for two reasons:
it is rotation-invariant, so it holds whatever body-axis convention the vehicle
adopts, and it is the quantity a pointing budget consumes directly —
one number that adds in quadrature with the control error. Splitting the
vehicle-level knowledge error per axis would be three numbers carrying the
information of one, so it is not done here; the axis-resolved form belongs to
the **per-sensor cross-boresight requirements** a payload brings with it, where
+Z is the sensor boresight and the split says something the norm cannot.

The thresholds below were **set from measurement, not from ambition**: each is
the value the verifying Monte Carlo campaign
(``tests/unit/attitude_accuracy_mc_test.cpp``) demonstrates with margin on the
reference vehicle's own sensor budget (``config/spacecraft/leo_smallsat.yaml``,
``flight.attitudeEstimator.*``). Improving them is a **sensor-budget** change —
magnetometer calibration, sun-sensor albedo correction, a star tracker — not an
estimator change; see the note on REQ-ADET-006.

.. _adet-campaign-method:

How the campaign is run
-----------------------

The verifying campaign is ``tests/unit/attitude_accuracy_mc_test.cpp``, and it
is what CI runs, so its numbers are the ones a degradation of the sensor budget
or of either estimator has to get past. Four choices in it decide what the
measured thresholds mean.

**Systematics are drawn as biases, not as noise.** Most of a coarse budget is
systematic — albedo on the sun sensor, IGRF model error and hard-iron residual
on the magnetometer, mount alignment — and the defining property of those terms
is that they are *the same offset every cycle*, so 10 Hz of fixes does not
average them away. Each run therefore draws its systematic offsets **once** and
holds them for the whole run, and draws the white terms **per measurement**.
Modelling the systematic part as white would make the campaign vacuously easy:
the reported error would fall as 1/√N of the samples instead of settling on the
floor the hardware actually has.

**Geometry is swept, not chosen.** The sun/field separation drives the TRIAD and
the filter alike, since the solution's component about the sun is fixed by the
magnetic pair alone and its error grows as 1/sin(separation). Runs draw the
separation uniformly across the **well-conditioned band, 45–135°** — the
condition the requirements are stated under — rather than sitting at the
orthogonal best case. Outside that band the 1/sin term takes over and the
distribution becomes a property of the observation geometry rather than of the
estimator: over the full range the flight gate admits (down to sin 10°) the same
campaign measures a 3σ bound near 19° in both modes, driven entirely by runs
below ~30° of separation. Near-parallel geometry is covered instead by the
refusal and coasting tests in ``coarse_attitude_test.cpp``, where the correct
behaviour is to decline to publish rather than to hold an accuracy.

**The statistical bound is distribution-free.** Each run contributes exactly one
sample — the error at the end of the run — so the samples are independent, and
the sample **maximum** over N runs is a one-sided upper confidence bound on the
99.73rd percentile at confidence :math:`1 - 0.9973^N`; N = 800 gives 88%.
Asserting on the maximum rather than on the empirical 99.73rd percentile is the
conservative choice, and it is the reason N is what it is.

**Each campaign also asserts a sensitivity floor** on its median. The upper
bound alone cannot fail when the noise is accidentally switched off, so a test
bug that zeroed the systematic draws would sail through it; the floor is what
catches that instead.

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
   :refs: markley2014, black1964, shuster1981, davenport1968

   The FSW **shall** seed the coarse solution and the MEKF from two or more
   vector measurements using a deterministic single-frame initializer (TRIAD /
   QUEST / q-method), giving a defined cold-start/acquisition path.

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

.. req:: Coarse-mode attitude-knowledge accuracy
   :id: REQ-ADET-005
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, accuracy, safe
   :method: Test
   :derived_from: REQ-ADET-002, REQ-MIS-004
   :allocation: lib/gnc, flight/components/AttitudeEstimation
   :value_required: <= 15 deg error norm (3-sigma)
   :margin_required: 20 %
   :refs: shuster1981

   In coarse mode, sunlit, with a valid sun-sensor, magnetometer and gyro set at
   the reference vehicle's sensor budget and a sun/field separation of **45° or
   more**, the attitude-knowledge error norm (:ref:`metric
   <adet-knowledge-metric>`) **shall** be ≤ **15°** (3σ).

   .. note::

      **Where the number comes from.** The floor is systematic, not statistical.
      The reference suite budgets a 2.04° 1σ per-axis systematic on the sun pair
      (albedo-dominated, ⊕ the analytic-ephemeris term) and 1.93° on the magnetic
      pair (uncalibrated hard iron), and a systematic offset is the *same* offset
      every cycle — 10 Hz of fixes does not average it down. Two offsets of that
      size put the median error norm at ~3.3°, and the 3σ point is set by the
      **Rayleigh tail** of the offset magnitudes (3.44σ = 7.0° for the sun pair
      alone). The verifying campaign measures a 3σ bound of **8.9°** over 800
      runs — 8.1–9.7° across the other master seeds tried while setting the
      threshold — so 15° carries 41% margin and does not sit on a tail that
      moves with the seed.

      **Why the geometry condition.** The solution's component about the sun is
      fixed by the magnetic pair alone and its error grows as
      1/sin(separation). Between the flight gate (sin 10°) and ~45° that term
      dominates and the same campaign measures ~19°, which is a property of the
      observation geometry rather than of the estimator. Below the gate the
      correct behaviour is to refuse the solve and coast, which REQ-ADET-002
      covers.

.. req:: Fine-mode attitude-knowledge accuracy, sun + magnetometer
   :id: REQ-ADET-006
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, accuracy, mekf
   :method: Test
   :derived_from: REQ-ADET-001
   :allocation: lib/gnc, flight/components/AttitudeEstimation
   :value_required: <= 15 deg error norm (3-sigma)
   :margin_required: 20 %
   :refs: markley2014

   In fine mode with the sun/magnetometer/gyro suite and no star tracker, under
   the same conditions as REQ-ADET-005, the attitude-knowledge error norm
   **shall** be ≤ **15°** (3σ).

   .. note::

      **The threshold equals the coarse one, and that is the finding.** With only
      two vector sources neither a fixed-gain complementary blend nor an optimal
      filter can separate a constant offset from the truth, so both settle on the
      same systematic floor — the design doc §8.1 caveat, quantified. The MEKF
      does buy the bulk of the distribution (a median of 2.9° against the
      coarse chain's 3.3° — a comparison of the two campaigns' distributions,
      not run-for-run: they draw from different master seeds, so the two medians
      are separate samples of the same budget rather than paired outcomes),
      gyro-bias observability, and a covariance that means something; it does
      not buy accuracy. The verifying campaign measures a 3σ bound of
      **11.4°**, giving 24% margin. Tightening this is a **sensor-budget** action
      — magnetometer hard-iron calibration and sun-sensor albedo correction are
      the two levers, worth ~2° each — or a star tracker (REQ-ADET-007). It is
      not an estimator-tuning action.

.. req:: Fine-mode attitude-knowledge accuracy with star tracker
   :id: REQ-ADET-007
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, accuracy, mekf, phase4
   :method: Test
   :derived_from: REQ-ADET-001
   :allocation: lib/gnc, flight/components/AttitudeEstimation
   :value_required: <= 0.05 deg error norm (3-sigma)
   :margin_required: 20 %
   :refs: markley2014

   In fine mode with one or more star trackers fused (§8.2), the
   attitude-knowledge error norm **shall** be ≤ **0.05°** (180 arcsec, 3σ).

   .. note::

      Not yet verifiable: the §8.2 multi-sensor fusion layer is unbuilt, so this
      requirement is held at ``reviewed`` and carries no verifying artifact. It
      is promoted to ``approved`` — the status the CI traceability gate acts on —
      by the push that lands the fusion layer and extends
      ``tests/unit/attitude_accuracy_mc_test.cpp`` with a star-tracker campaign.

      **Where the number comes from.** On the error-**norm** metric a single
      tracker is dominated by its about-boresight axis: for the AURIGA in
      ``config/hardware/star_tracker/sodern_auriga.yaml`` the Z terms are ~6×
      the cross-boresight ones (51 / 38 / 70 arcsec 3σ low-frequency spatial,
      high-frequency spatial, temporal — RSS ≈ 95 arcsec, against ≈ 16 arcsec
      cross-boresight), and the unit's fixed 61 arcsec bias (``bias_deg``
      0.017°) does not average down at all. That is ≈ 113 arcsec 3σ of hardware
      alone before installation
      alignment or fusion loss, which is why the threshold is 0.05° and not the
      ~0.02° a cross-boresight-only reading of the datasheet suggests. Two
      trackers with **non-parallel boresights** are what actually buy the
      0.02°-class number, since each covers the other's weak axis — a
      configuration decision this requirement deliberately does not presume.
