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
   ``SeedMinObservability`` from 0.0076 to **5.1e-4**. Both are named because the
   second is not optional: the Davenport seed gate is not invariant to changing
   the measurement sigmas *relative* to each other, so shipping the tightened
   sigma without re-deriving the gate refuses every geometry in the band. The
   5.1e-4 is derived in the post-fit uplink block of
   ``config/spacecraft/leo_smallsat.yaml`` and in design doc §8.1; it preserves
   the gate's 10°-separation meaning with the albedo correction *also* in force,
   which is the parameter set a calibrated vehicle is actually in.

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
   :value_required: <= 0.03 deg error norm (3-sigma), post-calibration parameter set and inter-tracker alignment in force
   :margin_required: 20 %
   :refs: markley2014

   In fine mode with **two star trackers fused** (§8.2) — the IMU still
   propagating the solution between updates — with the
   :ref:`post-calibration parameter set <adet-postcal-condition>` in force **and
   a valid inter-tracker alignment calibration (REQ-ADET-013)**, the
   attitude-knowledge error norm **shall** be ≤ **0.03°** (108 arcsec, 3σ).

   .. _adet-007-degraded-floor:

   **King-only is the degraded floor**, recorded rather than required, exactly as
   the analytic-ephemeris fallback is for REQ-ADET-006: before the alignment
   calibration has run, after ``ST_ALIGN_CAL_CLEAR``, or with the second unit
   latched out, the vehicle fuses the king alone and measures **0.024°**. That
   clears the 0.05° this requirement was written at but not 0.03° with the
   declared margin, which is precisely why the condition is carried rather than
   the threshold being loosened to cover both.

   .. note::

      **Enacted at 0.03° in Push 52, down from 0.05°.** Measured **0.021° 3σ,
      30% margin**. Measured by
      ``StarTrackerFineModeKnowledgeErrorNorm`` in
      ``tests/unit/attitude_accuracy_mc_test.cpp``, an 800-run campaign on the
      reference vehicle's two-AURIGA suite (boresights 90° apart, both 135° from
      the payload/array face) with the commanded inter-tracker alignment
      calibration flown through the real ``gnc::StAlignmentAccumulator`` before
      each run. Median 0.009°. The status stays ``reviewed`` for the
      repository-wide CI reason recorded below, not for any gap in the evidence.

      **The mode is ST+IMU only, and the wording above changed to say so.** The
      first form of this requirement said the tracker "joins the SS+MAG+IMU suite
      as a further vector source, not replacing it". That is not what the vehicle
      flies (user decision, 2026-08-02): with at least one valid tracker the sun
      and magnetic pairs are **not** fused at all. They are two orders of
      magnitude wider, so folding them in can only pull the solution away from the
      trackers, and the filter's white-``R`` model has no way to represent the
      systematic floor that makes them wide. They are not discarded either — they
      become FDIR-monitored residuals against the tracker solution
      (REQ-ADET-012), which is a strictly better use of them, because a sun sensor
      that has drifted is then *observable* instead of merely down-weighted.

      **Where the number comes from, and where the earlier estimate was
      pessimistic.** The pre-implementation note here read the AURIGA's
      about-boresight terms (51 / 38 / 70 arcsec 3σ low-frequency spatial,
      high-frequency spatial, temporal — RSS ≈ 95 arcsec, against ≈ 16 arcsec
      cross-boresight) and its fixed 61 arcsec bias as ≈ 113 arcsec of hardware
      alone. That added two 3σ figures as if they were the same kind of quantity.
      The bias is quoted as a **bound** on an isotropic offset, so its per-axis
      1σ is ``B/3`` = 20.4 arcsec, and the campaign's measured single-tracker
      bound is 0.024° (86 arcsec) rather than 113. That is also why the enacted
      threshold can be conditioned on two trackers rather than written to cover
      one: the single-tracker case is a real, reachable configuration, so it is
      carried as the :ref:`degraded floor <adet-007-degraded-floor>` instead of
      setting the threshold to the weakest configuration and giving away the
      margin the second unit buys.

      **What the second tracker actually buys, measured as a paired comparison.**
      ``SecondStarTrackerCoversTheFirstsWeakAxis`` flies the same run twice —
      identical truth, systematics and unit-0 noise sequence, the only difference
      being whether unit 1 is fused — so the comparison is paired rather than two
      independent samples. The dual configuration is better on **65.0% of 800
      runs**; under "the second tracker changes nothing" the win count would be
      Binomial(800, ½) = 50 ± 1.8%, so that is decisive. On the aggregate the
      bound moves 0.024° → 0.021° and the median gain is small.

      The datasheet asymmetry is ~6× on the *noise* terms, but the isotropic bias
      dominates the cross-boresight budget and compresses the ratio the fusion sees
      to 1.8×. The second unit does cover the first's weak axis — the algebra is
      pinned directly in
      ``Mekf.AnisotropicRIsWhatMakesTwoNonParallelTrackersWorthCarrying``, where
      the about-boresight variance falls by more than 10× — but on the *norm*
      metric that improvement is spent against a floor the trackers share.

      **That floor is the king tracker's own bias, and nothing removes it.** The
      king's mounting *defines* the body frame, so no alignment is estimated for
      it; the inter-tracker calibration (REQ-ADET-013) removes the *difference*
      between the two units, which is what lets them fuse without fighting. But
      the payload is mounted against the physical structure, not against the
      king's optical axis, so the king's 20.4 arcsec/axis bias is a real knowledge
      error. The campaign asserts the measured median stays *above* it, because a
      result below that floor would mean the campaign is averaging down a
      systematic that does not average down — the one way this number could be
      wrong in the flattering direction.

      **Why 0.03° and not the committed 0.02°.** §8.1 committed to 0.02° once a
      second non-parallel tracker landed. The measured bound sits *above* that at
      0.021°, so enacting 0.02° would have made this requirement fail outright,
      and anything near it would leave nothing like the declared 20% margin. A
      threshold at exactly 20% margin on the measured bound is 0.0264°; **0.03°**
      is that rounded to a number a requirement can carry, and it lands at 30%.

      Note how little separates 0.020° from 0.021°: the two are the same campaign
      before and after the RNG substreams were separated to make the single/dual
      comparison paired. A committed threshold that moves with a test-harness
      refactor is a threshold too close to its own measurement, which is the
      concrete reason 0.02° was not chased. Going below 0.03° needs the **king's
      own bias** reduced — a ground-calibrated tracker mounting, or an absolute
      alignment against a payload-derived reference — and neither is in the
      current design.

      The status stays ``reviewed`` rather than ``approved`` for the same reason
      every other requirement here does: the docs job that runs the
      ``req_without_verification`` gate does not execute the C++ suites, and the
      GoogleTest traceability artifact is not committed, so an ``approved``
      requirement would fail the gate on a run where its test never executed.

.. _adet-imu-voting:

Redundant-gyro combination
--------------------------

The reference vehicle carries **two** IMUs, and how they are combined is a
requirement rather than an implementation detail, because the obvious
combination is unsafe. The arithmetic mean has a **breakdown point of zero**
([rousseeuw1987] §1.2): one unit reporting an arbitrary value moves the mean by
an arbitrary amount. A gyro railed at full scale therefore does not degrade a
mean, it destroys it — and it does so while its own validity flag still reads
true, because a unit that has failed high has no way to know. Averaging converts
a survivable single-unit fault into a vehicle-level one, which is the opposite
of what redundancy is carried for ([gilmore1972]).

The two requirements below split that into the property (single-fault survival,
REQ-ADET-008) and the response (isolation, reporting and recovery,
REQ-ADET-009), because they fail independently: a combination could be robust
and silent, or loud and non-robust, and both are defects.

All three requirements added here (REQ-ADET-008 … REQ-ADET-010) **do** carry
verifying tests, unlike REQ-ADET-007 above, which cannot be verified until a
star tracker is fused. They are nonetheless held at ``reviewed`` rather than
``approved`` for the same reason every other requirement in this repository is:
the docs job that runs the ``req_without_verification`` gate does not execute
the C++ suites, and the GoogleTest traceability artifact
(``docs/_generated/verif_gtest.json``, produced by
``tools/dev/collect_gtest_trace.py``) is not a committed one — so an
``approved`` requirement would fail the gate on a run where its test never
executed. Promoting the baseline is a CI-pipeline change, not a per-requirement
decision, and it is owed to every requirement here at once.

.. req:: Single-fault-tolerant multi-IMU rate combination
   :id: REQ-ADET-008
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, fdir, redundancy
   :method: Test
   :derived_from: REQ-ADET-002, REQ-FDIR-002
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :refs: rousseeuw1987, gilmore1972

   With three or more IMUs reporting, the combined body rate **shall** be
   unaffected — beyond the spread of the healthy units themselves — by **one**
   unit reporting an arbitrary value, including a reading railed at full scale, a
   non-finite reading, and a stale one. The combination **shall not** be an
   arithmetic mean of the reporting units.

   With exactly two units reporting — the reference vehicle's configuration — a
   disagreement beyond the configured gate **shall** be detected, and **shall**
   be attributed to the offending unit using an independent rate reference (the
   fine-mode filter's propagated body rate) where one is available. Where none is
   available, the combined rate **shall** be reported invalid rather than
   published from an arbitrary choice of unit, leaving the estimator to hold
   attitude on a growing covariance.

   Attribution **shall** require a decisive comparison — the reference agreeing
   with one reading and disagreeing with the other — sustained over a configured
   number of consecutive cycles, so that neither an exact tie, a single noisy
   sample, nor a drift the reference shares with both units can latch a unit out.
   The reference itself **shall** be range-checked against the same physical rate
   limit the measurements are, and a failing reference treated as no reference.

   A disagreement that remains unattributable **shall** escalate after a
   configured continuous-ambiguity horizon, with a distinct event naming the
   duration, re-reported at a bounded cadence while the condition persists.

   .. note::

      The implementation (``lib/gnc/imu_voting``) is generic over N and carries
      the whole ladder — gated pass-through at one unit, pairwise detection and
      filter-based identification at two, a per-axis **median** at three or more
      (breakdown point ⌊(n−1)/2⌋/n). Which rung the vehicle flies is a
      configuration decision, and the reference vehicle's two IMUs put it on the
      pairwise rung. Parity-space isolation ([gilmore1972]) is the sharper
      instrument and is the upgrade path for a skewed or dissimilar array; it
      needs the array geometry, which this vehicle's configuration does not
      carry.

      **Verification, per rung.** The median rung: ``tests/unit/imu_voting_test.cpp``
      (including a sweep of the faulted unit's magnitude over four decades) and
      the component cases ``RailedImuIsExcludedAndCostsNothing`` and
      ``NonFiniteImuIsExcludedNotPropagated``, which assert the published
      attitude is **bit-identical** to the same run with three healthy units.
      The pairwise rung, which is the flown one:
      ``TwoImuDisagreementIsIdentifiedByTheFilter`` (identification, with the
      surviving unit carrying the solution) and
      ``TwoImuDisagreementLeavesNoRate`` (the conservative fallback with no
      filter available), both in
      ``flight/PolarisFsw/AttitudeEstimator/test/ut/``. On the vehicle,
      ``SitlImuVoting`` in ``tests/integration/sitl_attitude_tuning_test.cpp``
      injects a **plausible** 0.05 rad/s bias into one of the two units —
      about six times the disagreement gate, a tenth of the plausibility limit, so
      no per-unit gate can see it — and asserts the filter identified the
      offender (``OUTVOTED``, no ``ImuVoteAmbiguous``) and that the attitude
      solution and the fine mode both survive it.
      ``UnattributableDisagreementRefusesTheRateAndKeepsTheAttitude`` is its
      counterpart: the same fault applied before the filter has a rate to be
      believed against, asserting the refusal is reported once, that nothing is
      latched, and that the vehicle still acquires an attitude — the cost of the
      refusal is bounded to the body rate.

.. req:: IMU exclusion, reporting and re-admission
   :id: REQ-ADET-009
   :status: reviewed
   :level: L2
   :tags: adcs, fdir, redundancy, telemetry
   :method: Test
   :derived_from: REQ-FDIR-001
   :allocation: flight/PolarisFsw/AttitudeEstimator

   An IMU excluded from the rate combination **shall** raise an FDIR event naming
   the unit and the gate that closed, **shall** be latched out rather than
   re-entering on its next plausible sample, and **shall** be re-admitted only
   after a configured number of consecutive cycles **in which the unit passes the
   criterion that excluded it**, or by command.
   The set of excluded units **shall** be telemetered. A unit that is merely
   absent — stale or dropped out — **shall not** be latched out, because absence
   is not evidence of an implausible reading.

   .. note::

      "The criterion that excluded it" is not pedantry, it is the whole content
      of the clause. A unit excluded by a *gate* is judged on that gate; a unit
      excluded by *identification* was plausible by construction — it passed
      every per-unit gate and lost a comparison — so it is judged on agreement
      with the combination instead. Judging it on plausibility would re-admit it
      unconditionally and it would be outvoted again on the next cycle, flapping
      at the re-admission period with an FDIR event per lap, which is precisely
      the "one event on the transition" property this requirement asks for.

      The re-admission policy is automatic with hysteresis rather than
      command-only, because the dominant real cause of one implausible sample is
      transient (a bus glitch, a dropped frame, a shock past the modelled rate
      limit) and permanently spending a unit of redundancy on a transient is the
      more expensive error. The consecutive-cycle count is what stops a marginal
      unit flapping; a genuinely dead unit never accumulates it. ``RESET_ESTIMATOR``
      is the commanded path and clears every latch at once.

      Verified by ``tests/unit/imu_voting_test.cpp`` (policy, hysteresis and the
      absent-vs-implausible distinction) and by the component cases
      ``ExcludedImuIsReadmittedAfterRecovery``, ``StaleImuIsAbsentNotExcluded``
      and ``ResetClearsImuExclusions``; on the vehicle by ``SitlImuVoting``,
      which recovers the faulted unit mid-run and asserts the re-admission
      event. Identification (``OUTVOTED``) puts the loser under the same latch
      and policy as a gate failure, so there is one path for FDIR to reason
      about however the fault was found.

.. req:: Sun-sensor suite coverage and handoff
   :id: REQ-ADET-010
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, accuracy, redundancy
   :method: Test
   :derived_from: REQ-ADET-002, REQ-ADET-005
   :allocation: config/spacecraft, flight/PolarisFsw/AttitudeEstimator

   The sun-sensor suite **shall** place the Sun inside at least one unit's field
   of view for **every** vehicle attitude, and the attitude-knowledge accuracy of
   REQ-ADET-005 and REQ-ADET-006 **shall** hold, with their stated margin, across
   attitudes at which the selected unit changes.

   .. note::

      The reference vehicle satisfies the coverage clause with six GomSpace FSS
      units on the six body faces. Each 60° acceptance half-angle covers the
      solid-angle fraction (1 − cos 60°)/2 = 0.25 of the sphere, and six cones on
      the face normals cover it completely: the worst-placed direction is a body
      diagonal at arccos(1/√3) = **54.736°** from each of the three nearest
      normals. Dropping one face leaves 8.8% of the sphere — a region about the
      dropped normal — seen by nothing.

      The accuracy clause holds without new tuning, but on the **field edge**
      rather than that body diagonal, and the two are separate facts. The FSS is
      specified 0.5° (3σ) inside 45° incidence and 2.0° at the 60° field edge,
      and the shipped ``SigmaSunWhiteRad`` is the **edge** figure. The flight
      selector orders on *reported σ* with ties to the lowest index, and σ is a
      two-regime step, so under a tie it takes the lowest-indexed unit in view
      rather than the best-aimed one — which can sit anywhere out to the 60°
      edge. The edge figure bounds that; 54.736° would not. So 54.736° is a pure
      coverage-geometry fact (worst-case *best-available* incidence, a property
      of the layout alone) while the 60° field edge is the accuracy-bounding
      property of the shipped selector. Measured: mean best-unit incidence 31.9°,
      87.9% of directions inside the 45° regime, 47.3% seen by two or more units;
      worst selected incidence 60.0°, worst best-available 54.7°.

      Verified by ``SunSuiteCoversEveryAttitude`` (the coverage and worst-case
      incidence, checked directly) and ``KnowledgeHoldsAcrossSunSensorHandoffs``
      (the requirement campaign re-flown at slew rates through the suite, so the
      active unit changes several times inside each run) in
      ``tests/unit/attitude_accuracy_mc_test.cpp``, and by the component
      selection cases in ``flight/PolarisFsw/AttitudeEstimator/test/ut/``.
      Measured on the handoff sweep, with the campaign's selector mirroring the
      flight rule exactly: coarse 3σ bound **2.70°** against 5° (46% margin),
      fine **1.76°** against 3° (41% margin), with 708 of 800 runs handing off at
      least once and 1882 handoffs in total.

      **Deferred, and worth naming.** The measurement port already carries each
      sample's *realised* σ (§6.4), and the estimator does not consume it — it
      weights with the configured edge constant. Consuming it would tighten the
      fine solution wherever the Sun is near a boresight (88% of directions), and
      needs a per-cycle white-σ override on ``CoarseAttitudeInput``; the MEKF
      already takes σ per update.

.. _adet-mag-voting:

Redundant-magnetometer combination
----------------------------------

The reference vehicle carries **two** magnetometers (user decision, 2026-08-02),
and they are combined by the same argument the gyros are: the arithmetic mean has
a breakdown point of zero ([rousseeuw1987] §1.2) whatever it is averaging, so a
second magnetometer averaged in is a second way to lose the field rather than a
redundancy. What differs is the *physical gate* — a magnetometer has a natural
one that a gyro does not — and the *cost of refusing*, which is one of two vector
pairs rather than the body rate.

.. req:: Fault-tolerant multi-magnetometer field combination
   :id: REQ-ADET-011
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, fdir, redundancy
   :method: Test
   :derived_from: REQ-ADET-002, REQ-FDIR-002
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :refs: rousseeuw1987, gilmore1972

   Each magnetometer's reading **shall** be gated on its magnitude against the
   onboard IGRF-modelled field magnitude at the vehicle's current position, within
   a configured ratio band, before it enters any combination. A cycle with no
   modelled field **shall** produce no combined field rather than a combination
   whose only surviving gate is finiteness.

   The combined field **shall not** be an arithmetic mean of the reporting units.
   With three or more units reporting it **shall** be unaffected — beyond the
   spread of the healthy units themselves — by one unit reporting an arbitrary
   in-band value. With exactly two units reporting, a disagreement beyond the
   configured gate **shall** be detected, and **shall** be attributed to the
   offending unit using the modelled field rotated into body axes where the
   attitude solution behind that rotation meets a configured **quality** bound,
   not merely a validity flag. Where no such reference is available the combined
   field **shall** be reported invalid rather than published from an arbitrary
   choice of unit.

   An excluded magnetometer **shall** raise an FDIR event naming the unit and the
   gate that closed, **shall** be latched out, and **shall** be re-admitted only
   after a configured number of consecutive cycles passing the criterion that
   excluded it, or by command. The set of excluded units **shall** be
   telemetered, and a unit that is merely absent **shall not** be latched out.

   An unattributable disagreement **shall** escalate after a configured
   continuous-ambiguity horizon, with a distinct event naming the duration,
   re-reported at a bounded cadence.

   .. note::

      **The plausibility gate is the interesting clause**, and it is what makes
      this requirement different from REQ-ADET-008 rather than a copy of it. The
      vehicle already evaluates IGRF-14 at its own position every cycle to build
      the magnetic reference, so ``‖m‖`` against ``‖B_IGRF‖`` costs nothing new
      and is *attitude-free* — it works in Safe mode and at cold start. It is
      strictly better than a fixed full-scale check: it tracks the field from
      ~22 µT to ~52 µT over an orbit instead of admitting everything below
      saturation, so a reading that is plausible at one point in the orbit is
      correctly rejected at another. It is stated as a **ratio** so it does not
      need re-deriving when the orbit changes.

      **The reference's quality condition is the second one.** A validity flag
      cannot tell a 0.5° solution from a 10° one, and a 10° attitude error
      mispredicts a 30 µT field by ~5 µT — the disagreement gate itself. Gating
      on validity alone would hand the verdict to whichever unit happened to sit
      nearer a badly rotated prediction, which is the same failure the IMU vote's
      range-checked reference guards against, one level up.

      **Refusing is cheaper here than for the gyros, and that is by design.** An
      unattributable magnetometer disagreement costs the magnetic pair for that
      cycle — the coarse chain refuses TRIAD and the MEKF skips the magnetic
      update, which is the estimator's existing dropout behaviour — where the IMU
      equivalent costs the body rate itself. The vehicle keeps propagating and,
      with the Sun in view, keeps acquiring. The escalation horizon is *shared*
      with the IMU vote (``ImuAmbiguityEscalateCycles``) rather than duplicated:
      how long the ground should wait before a persistent refusal reaches a
      console is a property of the operations concept, not of the sensor.

      **Implementation.** The policy — plausibility ladder, median at three or
      more, pairwise identification with margin and confirmation gates, exclusion
      latch, criterion-matched re-admission — is shared with the multi-IMU vote in
      ``lib/gnc/unit_voting``; ``lib/gnc/mag_voting`` is the thin wrapper that
      supplies the two magnetometer-specific gates above. That sharing is the
      point: a divergence between the two votes' policies would be a defect, and
      one implementation cannot diverge from itself.

      Verified by ``tests/unit/mag_voting_test.cpp`` (14 cases mirroring the IMU
      suite's shapes, including the absurd-reference, indecisive-comparison,
      flipping-verdict, flap and re-admission ones, plus the band cases the IMU
      suite has no analogue for) and by the component cases
      ``ImplausibleMagnetometerIsExcludedAndCostsNothing`` and
      ``TwoMagnetometerDisagreementLeavesNoMagneticPair`` in
      ``flight/PolarisFsw/AttitudeEstimator/test/ut/``. The first asserts the
      published attitude is unchanged by a dead unit; the second asserts the cost
      is bounded to the magnetic pair, with the body rate still flowing.

.. _adet-mode-ladder:

The estimation mode ladder
--------------------------

.. req:: Estimation mode ladder and residual monitoring of demoted sources
   :id: REQ-ADET-012
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, fdir, mekf
   :method: Test
   :derived_from: REQ-ADET-004
   :allocation: flight/PolarisFsw/AttitudeEstimator
   :refs: markley2014

   The fine estimator **shall** select its measurement sources by a fixed ladder:
   with at least one valid star-tracker solution, star trackers **only**;
   otherwise the sun and magnetic vector pairs; with neither, the coarse chain is
   the published product. The active rung **shall** be telemetered, and
   transitions between rungs **shall** be surfaced as events.

   A rung transition **shall not** be a demotion: the filter **shall** retain its
   state and covariance across it, and the published solution **shall** remain
   valid, so that a consumer needing only *an* attitude sees no interruption while
   a consumer with a knowledge requirement can gate on the rung.

   A star tracker whose solutions the filter's consistency gate rejects on a
   configured number of consecutive cycles **shall** be excluded from the fusion
   and reported, **without demoting the mode**, and **shall** be re-admitted after
   a configured run of cycles agreeing with the fine solution, or by command. The
   mode-level rejection streak **shall not** advance on a cycle where any tracker
   was accepted.

   On any cycle where the sun or magnetic pair is **not** being fused, its
   disagreement with the fine solution **shall** be computed, telemetered, and
   monitored: a residual past a configured threshold for a configured number of
   consecutive cycles **shall** raise an FDIR event naming the source, re-reported
   at a bounded cadence, and cleared when it returns inside the threshold.

   Where two or more sun sensors report the Sun in view, the selected unit's
   direction **shall** be cross-checked against the runner-up's; a persistent
   disagreement **shall** raise an FDIR event, and where a fine solution is
   available to judge with, the estimator **shall** use whichever of the two
   agrees better with it.

   .. note::

      **Why ST-only rather than ST-plus-vectors.** The sun and magnetic pairs are
      two orders of magnitude wider than a tracker, so fusing them alongside can
      only pull the solution away from the trackers; and the filter treats ``R``
      as white, so it has no way to represent the systematic floor (albedo
      residual, IGRF model error, hard iron) that makes them wide — it would
      average down an offset that does not average down and report a covariance
      tighter than the truth for the privilege.

      **What demoting them buys is not nothing — it is observability.** A source
      the filter is updating from cannot be checked against the filter: the
      residual is small *because* the update made it small. Demoted, the same
      measurement becomes a monitor, and a sun sensor that has drifted 20° is now
      an event on the ground's console instead of a slightly worse solution. The
      component test ``DriftedSunSensorRaisesTheResidualMonitor`` is exactly this
      claim: the alert fires and the published attitude does not move.

      **The sun cross-unit check closes a gap the selector leaves open.** The
      flight rule takes the smallest *reported* σ, so a unit that is confidently
      wrong — small σ, wrong direction — wins, and nothing else on the vehicle
      would notice. Detection needs no attitude at all (it is one sensor against
      another), which makes it the only cross-check available in Safe mode;
      resolution needs one, and is gated on the monitor having already alerted so
      a single noisy sample can never move the selection. Nothing is latched: the
      override is re-decided each cycle from the current evidence, so a unit that
      recovers simply stops being overridden and there is no exclusion for the
      ground to reason about on a condition the vehicle resolved itself.

      **A monitor that cannot run does not clear its own alert.** A cycle where
      the source is absent resets the streak but leaves an existing alert
      standing, because "we stopped looking" is not "it recovered" — otherwise a
      faulted sensor could close its own alert by dropping out.

      **A bad tracker is isolated; the mode is not.** A cycle-global rejection
      streak ORs every tracker's verdict together, so one persistently-disbelieved
      unit demotes the whole fine mode — which drops the filter, re-promotes off
      the *same* bad unit, and flaps at the streak period, roughly 0.5 Hz on the
      reference tuning. The unit is the thing to isolate, exactly as a disagreeing
      IMU or magnetometer is. So the streak is **per unit**, and the mode-level one
      is suppressed by any tracker acceptance: "the filter believes nothing it is
      being told" is the condition that actually means divergence.

      The vector path keeps the Push 44 rule unchanged, and the asymmetry is
      deliberate: with no tracker fused, isolating one of the two vector sources
      would leave a filter running on a single wide source, whereas isolating one
      tracker leaves it running on an arcsecond-class one.

      Re-admission is judged on the criterion that excluded it — agreement with the
      solution the *surviving* trackers built — at the unit's own 3σ about its weak
      axis, not at one of the residual-monitor thresholds. Those are sized for a
      sun sensor or a magnetometer, in degrees, and an arcsecond-class instrument
      that is degrees out would sail through one and earn its way back while still
      grossly wrong, which is how a re-admission policy quietly becomes a no-op.

      Verified by ``BadStarTrackerIsIsolatedWithoutDemotingTheMode`` (the
      regression for exactly this: the king healthy, the second unit 5° out, the
      mode staying engaged on the king with the bad unit latched out and later
      re-admitted), ``StarTrackerTakesTheLadderToItsTopRung``,
      ``StarTrackerLossFallsBackToSunAndMagnetometer``,
      ``DriftedSunSensorRaisesTheResidualMonitor``,
      ``MissingStarTrackerTuningCapsTheLadder`` and
      ``SunCrossUnitCheckAlertsAndOverrides`` in
      ``flight/PolarisFsw/AttitudeEstimator/test/ut/``.

.. req:: Commanded inter-star-tracker alignment calibration
   :id: REQ-ADET-013
   :status: reviewed
   :level: L2
   :tags: adcs, estimation, calibration, mekf
   :method: Test
   :derived_from: REQ-ADET-007
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :refs: markley2007, markley2014

   One star tracker **shall** be designated the **king**: its mounting defines the
   vehicle body frame, no alignment **shall** be estimated for it, and every other
   tracker's solution **shall** be stated in the king's frame before it reaches
   the estimator.

   The vehicle **shall** provide a commanded on-orbit calibration that estimates a
   non-king tracker's constant rotation relative to the king from **simultaneous**
   solution pairs — start, abort and clear commands; a window counted in accepted
   pairs with a bounded self-close deadline; a fit on **uncorrected** readings; a
   single application point; and the fit's residual, the misalignment it found and
   the set of units carrying a correction telemetered.

   Every unobservable or untrustworthy case **shall** be a refusal that applies
   nothing and retains any previously applied correction, reporting the gate that
   closed: too few pairs, pairs that do not share one fixed rotation, excessive
   dispersion, numerical failure, missing tuning, and a commanded unit that is the
   king, out of range, or has no configured boresight.

   .. note::

      **Naming a king removes an unobservable degree of freedom rather than
      hiding one.** Two trackers on a structure observe only their *relative*
      rotation: an unmodelled common rotation of the whole assembly is
      indistinguishable from a rotation of the body frame, so estimating two
      absolute alignments from tracker data alone is estimating six parameters
      from three observable ones. What this does **not** do is remove the king's
      own bias — that is a real knowledge error against the physical structure the
      payload is mounted to, and it is the floor REQ-ADET-007 measures.

      **The estimator is the maximum-eigenvalue quaternion average**
      ([markley2007]): the eigenvector of ``M = Σ q_rel q_relᵀ`` for its largest
      eigenvalue. Arithmetic-mean-then-renormalise is only its first-order
      approximation and biases with dispersion; the outer product is invariant to
      the ±q sign ambiguity, so no sample can cancel another by having been
      reported with the opposite sign. ``M`` is a 4×4 accumulator, so the window
      is **O(1) in its length** — the same property the magnetometer calibration's
      normal equations have, and for the same reason: a flight component must not
      hold a window's worth of samples.

      **The quality metrics come out of the same eigenvalues, exactly.** With
      ``λ_max = Σ cos²(θᵢ/2)``, the RMS residual is ``2·sqrt(1 − λ_max/N)``
      without revisiting a sample, and the normalised gap ``(λ_max − λ₂)/N`` is
      the consistency metric.

      **The eigen-gap gate is not a geometry gate, and saying so matters.**
      Attitude pairs have no degenerate geometry — one pair determines all three
      parameters, at any attitude — so there is no analogue of TRIAD's
      near-parallel refusal here and no coverage gate to get wrong. A window taken
      with the vehicle parked is as valid as one taken through a tumble
      (``StAlignment.FitsFromASingleAttitude`` pins this). What the gap detects is
      a *fault*: a unit delivering solutions that do not sit at a fixed rotation
      from the king's — a mis-identified star field, a stale or cross-wired
      solution, a mounting that is moving. Refusing there is refusing to average a
      rotation that does not exist.

      **Simultaneity is the measurement.** A pair enters only on a cycle where
      both units delivered a fresh valid solution; at 0.1 °/s a one-cycle skew is
      already 36 arcsec, comparable to what is being measured. A cycle where only
      one unit solved costs the window a pair, not its correctness — which is why
      the window is counted in pairs and carries a self-close deadline, exactly as
      the magnetometer window is counted in accepted samples.

      **There is deliberately no "must beat the uncalibrated fit" gate** of the
      kind the magnetometer calibration carries. An ellipsoid fit can converge on
      a worse sensor model; an alignment estimate is a mean of a quantity that is
      either constant (and the mean is right) or not (and the eigen-gap gate
      catches it). A comparison against "no correction" would only ever fire where
      the true misalignment is smaller than the noise, and applying the estimate
      there is harmless.

      **The correction does not survive a reboot**, on the same deferral as the
      magnetometer calibration: it lives in component state and nothing is written
      to ``ParameterDb`` (§23.6 owns non-volatile state). A vehicle whose event log
      carries no ``StAlignComplete`` is fusing its second tracker as mounted.

      Verified by ``tests/unit/st_alignment_test.cpp`` (exact recovery of a known
      misalignment including the composition order, the 1/√N averaging of the
      units' own noise against a residual that correctly does *not* fall with N,
      the single-attitude case, and each refusal) and by the component cases
      ``InterTrackerAlignmentCollectsFitsAndApplies``,
      ``InterTrackerAlignmentRefusesTheKingAndBadCommands`` and
      ``InterTrackerAlignmentAbortAndClear``. In the REQ-ADET-007 campaign the fit
      is flown on every one of the 800 runs
      (``InterTrackerAlignmentFitsOnEveryRun``): 800/800 accepted, residual median
      45.4 arcsec against the shipped 103 arcsec gate, misalignment median
      44.8 arcsec.

      **A non-king tracker is not fused until this calibration has run.** Its
      as-mounted reading carries the two units' bias *difference* — 45-110 arcsec
      measured — and fusing it at the ~21.5 arcsec σ the configuration declares
      would sell a systematic as white noise, which is the overconfidence the
      white-``R`` model cannot represent. So the vehicle flies king-only until the
      window closes, which is also the honest launch state and what makes
      ``ST_ALIGN_CAL_CLEAR`` a safe command rather than one that silently degrades
      the solution. Pinned by ``UncalibratedSecondTrackerIsNotFused``.

.. req:: Fine-mode filter usability — retune without loss, covariance re-initialisation, selective processing
   :id: REQ-ADET-014
   :status: reviewed
   :level: L3
   :tags: adcs, estimation, mekf, operability, fdir
   :method: Test
   :derived_from: REQ-ADET-004
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator
   :refs: carpenter2018, dennehy2020

   The fine-mode attitude filter (MEKF) **shall** implement the NESC
   navigation-filter usability practices of NASA/TP-2018-219822 Ch. 9 and NESC
   Technical Bulletin 20-03 items (d), (f) and (g):

   * a fine-mode parameter upload **shall** re-tune the running filter in place
     — attitude, gyro bias, covariance and age kept, no demotion — and a set that
     fails validation **shall** leave the last valid set in force (TP §9.3);
   * ``ATT_REINIT_COV(attSigma, biasSigma)`` **shall** re-open the covariance
     around the current attitude and bias without altering either (TP §9.2);
   * a three-way ``ACCEPT`` / ``INHIBIT`` / ``FORCE`` policy **shall** be
     uplinkable per measurement type (sun vector, magnetic field, star-tracker
     attitude); ``INHIBIT`` **shall** withhold the type from **both** the coarse
     and the fine chain, ``FORCE`` **shall** apply it to the fine filter past its
     NIS gate, and forced updates **shall** be counted apart from acceptances and
     rejections (TP §9.1). ``FORCE`` **shall not** override a numeric fault.

   The fine covariance **shall** be checked for positive semi-definiteness every
   cycle (TP Ch. 7) and an indefinite covariance reported by event.

   Rationale: before Push 71 **any** parameter upload to the estimator — a
   star-tracker sigma, an albedo term — rebuilt the MEKF and demoted to coarse,
   throwing away a converged gyro-bias estimate to change a number that had
   nothing to do with it; and the only recovery from an over-confident fine
   filter was ``RESET_ESTIMATOR``, which loses the bias too. Underweighting (TP
   Ch. 4) is not adopted: the star-tracker measurement is linear in the error
   state (``H = [I 0]``), and for the sun/magnetic vector measurements the
   second-order term ``½ tr(H_kk P)`` is ≈ δθ²/2 ≈ 1e-3 rad at a 3° post-coast
   error against a 1e-2 rad measurement σ — an order below the noise it would
   be compensating.

   Verified by ``tests/unit/mekf_test.cpp``
   (``Mekf.RetuneKeepsTheSolutionAndRefusesABadConfig``,
   ``CovarianceReinitialisationKeepsTheState``,
   ``ForceOverridesTheGateAndIsCountedApart``) and the ``AttitudeEstimator``
   component tests ``FineTuningUploadKeepsTheSolution`` and
   ``FineCovarianceReinitAndMeasurementPolicy``.

.. req:: Fine-mode filter fidelity — order-invariant update, covariance reset, bias model, tracker latency
   :id: REQ-ADET-015
   :status: reviewed
   :level: L3
   :tags: adcs, estimation, mekf, star_tracker
   :method: Test
   :derived_from: REQ-ADET-004, REQ-ADET-007
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator, sim/sensors
   :refs: carpenter2018, reynolds2008

   The fine-mode attitude filter **shall** implement the measurement-processing
   and attitude-estimation practices of NASA/TP-2018-219822 Ch. 3, Ch. 5 and
   Ch. 8 that its structure allows:

   * **Order-invariant same-epoch update** (TP §3.2, Algorithm 3.1): all
     measurements of one GNC epoch **shall** be linearised at the same reference
     attitude, each innovation taken against the corrections already
     accumulated, and the multiplicative reset applied **once** per epoch, so
     the result does not depend on the order the trackers and the vector pairs
     are processed in. A propagation **shall** close an open batch first.
   * **Reynolds covariance reset** (TP Eq. 8.76): on each reset the attitude
     covariance **shall** be re-expressed in the corrected frame,
     ``P ← (I − [δθ̂×]/2) P (I − [δθ̂×]/2)ᵀ``, unless configured off.
   * **Gauss-Markov gyro bias option** (TP §5.2.4): a configurable correlation
     time ``MekfBiasTauSec`` **shall** select a first-order Gauss-Markov bias
     model with bounded variance ``σ_u² τ/2``; zero **shall** keep the random
     walk. The reference vehicle flies zero (its IMU's turn-on bias is
     constant; TP §5.2.7).
   * **Star-tracker latency** (TP §3.1): the truth model **shall** deliver a
     tracker's solution ``latency_s`` after the frame it describes, tagged at
     that frame's epoch (a delay line, as the receiver's fix latency is), and
     the filter **shall** advance a solution tagged behind the epoch on its own
     bias-corrected rate before comparing it, inflating ``R`` by the rate error
     integrated over the latency; a latency without a usable rate **shall** be
     refused. The staleness window ``MaxMeasAgeSec`` **shall** exceed the
     installed trackers' catalogued latency (configc-enforced flight/sim pair).

   Rationale: at the 0.5 °/s slew limit a 100 ms tracker latency is 0.87 mrad,
   ten times the AURIGA's cross-boresight σ — uncompensated, a slewing vehicle
   gates every tracker sample and falls to the sun/magnetic rung (measured in
   ``Mekf.LatentAttitudeMeasurementIsAdvancedOnTheFilterRate``: NIS above the
   gate uncompensated, innovation < 1e-6 rad compensated). The TP records the
   order dependence of sequential resets as a known divergence mechanism when a
   large prior error meets a precise measurement, and Reynolds found the
   covariance reset speeds convergence on exactly the large-update case
   re-acquisition presents.

   Verified by ``tests/unit/mekf_test.cpp``
   (``SameEpochBatchIsInvariantToMeasurementOrder``,
   ``ReynoldsCovarianceResetRotatesPByHalfTheCorrection``,
   ``GaussMarkovBiasIsBoundedAndDecaysAtTheCorrelationTime``,
   ``LatentAttitudeMeasurementIsAdvancedOnTheFilterRate``),
   ``tests/unit/sim_sensors_star_tracker_test.cpp``
   (``LatencyDeliversTheEarlierSolutionTaggedAtItsOwnEpoch``), the
   ``AttitudeEstimator`` component tests
   ``StarTrackerLatencyIsCompensatedOnTheFilterRate``,
   ``GaussMarkovBiasOptionIsAcceptedAndBounded`` and
   ``SameEpochBatchIsClosedBeforeThePublish`` (a 2° truth step is closed by the
   same cycle's pair before the product is published), the configc pair test
   ``test_staleness_window_under_the_tracker_latency_is_refused``, the SITL row
   ``SitlAttitudeControl.LatentStarTrackerStaysFusedThroughASlewAtTheRateLimit``
   (a 30° slew at the 0.5 °/s limit on the delayed AURIGA: tracker never
   excluded, source never falls to SUN_MAG, no demotion, at most one re-seed),
   and — with the AURIGA entry now carrying ``latency_s: 0.1`` — every
   star-tracker row of ``tests/integration/sitl_fault_matrix_test.cpp``, which
   flies the delayed tracker end to end.
