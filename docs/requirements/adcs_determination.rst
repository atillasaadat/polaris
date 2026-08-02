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

evaluated in code as :math:`2\,\mathrm{atan2}(\|q_{\mathrm{err},v}\|,
|q_{\mathrm{err},0}|)` — the same angle, but well conditioned at the small
errors the requirements are actually met at, where the scalar part alone has
lost half its significant digits (see ``lib/README.md``) —
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
   :allocation: flight/PolarisFsw/AttitudeEstimator
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
   :allocation: flight/PolarisFsw/AttitudeEstimator
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
   :allocation: flight/PolarisFsw/AttitudeEstimator

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
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :value_required: <= 5 deg error norm (3-sigma), post-calibration parameter set in force
   :margin_required: 20 %
   :refs: shuster1981

   In coarse mode, sunlit, with a valid sun-sensor, magnetometer and gyro set at
   the reference vehicle's sensor budget, a sun/field separation of **45° or
   more**, and **the post-calibration parameter set in force**, the
   attitude-knowledge error norm (:ref:`metric <adet-knowledge-metric>`)
   **shall** be ≤ **5°** (3σ).

   .. _adet-postcal-condition:

   **The post-calibration parameter set** means: a magnetometer hard/soft-iron
   calibration has been fitted on orbit (``MAG_CAL_START``) and is applied, the
   sun-vector albedo correction is applied, and the ground has **uplinked** the
   two parameters the fit invalidates —
   ``flight.attitudeEstimator.SigmaMagSysRad`` from 0.0337 to ~1.5e-3 rad, and
   ``SeedMinObservability`` from 0.0076 to ~1.1e-4. Both are named because the
   second is not optional: the Davenport seed gate is not invariant to changing
   the measurement sigmas *relative* to each other, so shipping the tightened
   sigma without re-deriving the gate refuses every geometry in the band.

   **As delivered the vehicle does not meet this**, and that is stated rather
   than hidden. ``config/spacecraft/leo_smallsat.yaml`` ships the uncalibrated
   values, so a freshly-commissioned vehicle sits on the uncalibrated floor —
   **8.49°** coarse and 8.07° fine, measured by the same campaign — until the
   calibration is flown and its parameters uplinked. That is a commissioning
   step, not a defect: the calibration is by design commanded, executed and
   assessed on orbit (§8.1), because a hard-iron signature is a property of the
   integrated vehicle and cannot be known before flight.

   .. note::

      **Where the number comes from.** The floor is systematic, not statistical.
      A systematic offset is the *same* offset every cycle — 10 Hz of fixes does
      not average it down — so the 3σ point is set by the **Rayleigh tail** of the
      offset magnitudes rather than by any noise average.

      The threshold was **tightened from 15° to 5° once the two calibration items
      the original note named had landed and run on the vehicle**: the on-orbit
      magnetometer hard/soft-iron calibration (§8.1, 1.93° → 0.087°) and the
      sun-vector Earth-albedo correction (2.04° → 0.72° per axis). The verifying
      campaign measures a 3σ bound of **2.48°** over 800 runs on the
      post-calibration budget, so 5° carries **50% margin**, and that margin is
      itself CI-enforced rather than merely reported. On the analytic-ephemeris fallback the
      bound is 2.77°, still clearing 5° with 45% margin — the coarse threshold is
      therefore stated flat, unlike REQ-ADET-006's.

      For the record of what the *uncalibrated* suite delivers — the as-delivered
      state above — a 2.04° per-axis sun systematic and 1.93° magnetic put the
      median at ~3.3° and the bound at **8.49°**, which is what the superseded
      15° threshold was set against. The campaign still runs that budget as the
      baseline its projections measure from.

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
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :value_required: <= 3 deg error norm (3-sigma), post-calibration parameter set in force and DE440 ephemeris tables active
   :margin_required: 20 %
   :refs: markley2014

   In fine mode on the same SS+MAG+IMU suite as REQ-ADET-005 and with no star
   tracker, under the same conditions — including **the post-calibration
   parameter set in force** (:ref:`as defined above <adet-postcal-condition>`) —
   **and with the onboard DE440 sun-ephemeris tables active** (serving grade
   ``kPrecise``), the attitude-knowledge error norm **shall** be ≤ **3°** (3σ).
   Both modes use all three sensor types; what differs is the estimator — a
   fixed-gain complementary blend against optimal gains with gyro-bias states.

   This requirement therefore carries **two** conditions, and they are not equal
   in weight. The **parameter-set condition is the larger dependency**: it needs
   an on-orbit calibration campaign, a successful fit, and a ground uplink, and
   without it the vehicle sits at 8.07° — nearly three times this threshold. The
   ephemeris condition needs only a table upload, is worth 0.44° at the bound,
   and the vehicle carries both grades and selects between them per cycle
   unaided. Losing the tables degrades; never having calibrated does not meet.

   Operating on the **analytic sun-ephemeris fallback** — no table upload in
   place, or the epoch past the end of the uploaded span — is a documented
   degraded mode, not a violation of this requirement. The measured bound there
   is **2.69°**, which still clears 3° but at only ~10% margin against the 20%
   required above; that gap is the whole reason the threshold is conditioned.

   .. note::

      **Why the threshold is conditioned rather than flat.** Stating one
      unconditioned number would force a choice between a threshold the vehicle
      misses whenever an upload lapses and one that gives away the margin an
      upload buys. The estimator already distinguishes the two cases *per cycle*
      — it composes the sun systematic as ``hypot(albedo term, ephemeris term)``
      with the ephemeris term selected by the served ``TableGrade`` (§8.1, Push
      48) — so the requirement can distinguish them too. Which term was in force
      on any cycle is reconstructible from downlinked telemetry:
      ``OnboardTables.EphemGrade`` publishes the served grade and the estimator's
      EPHEMERIS-domain ``ReferenceDegraded``/``ReferenceRecovered`` events mark
      every transition.

      **Where the number comes from.** With the tables active the verifying
      campaign measures a 3σ bound of **2.25°**, giving **25% margin**. The sun
      budget dominates what is left: 10.5 mrad of albedo residual against an
      ephemeris term that all but vanishes at grade ``kPrecise``. Tightening
      further is a sensor-budget action — an onboard reflectivity model attacks
      the albedo dispersion, a star tracker (REQ-ADET-007) attacks everything —
      not an estimator-tuning one.

      **The fine and coarse thresholds are no longer equal, and that is the
      change.** Uncalibrated they were, and that was the finding: with only two
      vector sources neither a fixed-gain blend nor an optimal filter can
      separate a constant offset from the truth, so both settled on the same
      systematic floor (8.1° against 8.5°, better by 0.4° on a floor of 8°). Once
      the calibration items removed most of that floor the filter's advantage
      became visible in the threshold — 2.25° against 2.48° — because what is
      left is much closer to the white-noise regime a Kalman gain is optimal in.

      **The MEKF does win, run for run.** Both chains are driven by the *same*
      draw — geometry, truth motion, systematic biases and every noise sequence —
      so the comparison is paired rather than two samples of the same budget.
      Measured on the uncalibrated baseline, where the systematic floor dominates
      and the comparison is cleanest, the filter is the more accurate of the two
      on **65% of the 800 runs**, with a median per-run improvement of 0.20° and
      a better bound at both the median (2.9° against 3.3°) and the tail. Under
      "the two are equally good" the win count would be Binomial(800, ½), i.e.
      50 ± 1.8%, so 65% is not sampling noise; the win fraction sits between
      64.2% and 65.4% across every master seed tried. The filter also buys
      gyro-bias observability and a covariance that means something, neither of
      which this metric can see.

      What it cannot buy is a different floor — that came from the sensor budget,
      and both levers have now been pulled: the magnetometer calibration and the
      albedo correction were worth ~2° each, as predicted, and together with the
      DE440 ephemeris grading they are what moved this threshold from 15° to 3°.
      The remaining lever is a **star tracker** (REQ-ADET-007), not estimator
      tuning.

.. req:: Fine-mode attitude-knowledge accuracy with star tracker
   :id: REQ-ADET-007
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, accuracy, mekf, phase4
   :method: Test
   :derived_from: REQ-ADET-001
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :value_required: <= 0.05 deg error norm (3-sigma)
   :margin_required: 20 %
   :refs: markley2014

   In fine mode with one or more star trackers fused (§8.2) — the star tracker
   joining the SS+MAG+IMU suite as a further vector source, not replacing it,
   with the IMU still propagating the solution between updates — the
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
