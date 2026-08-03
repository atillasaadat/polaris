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
