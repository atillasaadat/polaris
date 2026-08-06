Attitude Control (ACTL)
=======================

Attitude control law requirements. Source: design doc §8.5 (control), §7
(actuators and the MTQ/MAG duty-cycle interlock), §9 (FDIR).

.. req:: B-dot detumble
   :id: REQ-ACTL-001
   :status: reviewed
   :level: L2
   :tags: adcs, control, detumble
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController
   :refs: avanzini2012, markley2014

   The FSW **shall** provide a B-dot detumble control law commanding the
   magnetorquers from the measured rate of change of the geomagnetic field.

   The field derivative **shall** be formed from **successive magnetometer
   samples and their own time tags**, never from the nominal control period, and
   samples whose spacing falls outside a configured band **shall** be refused
   rather than used. The commanded dipole **shall** be scaled so that the
   *average* dipole over a control period — the commanded value times the §7
   duty factor — is the value the control gain demands, and **shall** be
   saturated per rod.

   From a 5 deg/s tip-off rate the law **shall** reduce the body-rate magnitude
   below **3.8 deg/s within 200 s of engaging**, and the rate **shall not**
   subsequently rise above that bound while the law is active.

   The window opens at *engagement*, not at boot, because B-dot's only input is
   the voted magnetometer field and that field does not exist until the vehicle
   has a position to evaluate the onboard IGRF at — the reference vehicle's GNSS
   receiver quotes a 34 s cold start. Charging the control law for the receiver's
   datasheet would make the requirement a statement about the wrong component.

   *Why the bound is where it is.* A B-dot law damps only the body-rate
   components perpendicular to the field: the component along it produces no
   ``dB/dt`` in body axes and is invisible to the law. What eventually removes it
   is the field direction turning over the orbit, a ~1e-3 rad/s process against a
   spin of ~5e-2 rad/s, so the vehicle settles into a slow spin about the local
   field line and unwinds it over **orbits**, not minutes — the asymptotic
   convergence Avanzini & Giulietti prove, not a defect. The requirement is
   therefore written on the fast phase, which is the one that decides
   controllability; the reference vehicle measures 3.14 deg/s 200 s after
   engaging, so the 3.8 deg/s bound carries 21 % margin. **Owed:** a Monte Carlo campaign over
   several orbits to characterise the residual-spin tail and set a Safe-mode
   handover time. The ``DetumbleExitRadps`` completion predicate exists and is
   telemetered, but no requirement is written on the time to reach it until that
   campaign has run.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``DetumblesFromFiveDegreesPerSecond``), with the law's dissipativity — the
   rotational kinetic energy non-increasing on every cycle it commands, saturated
   or not — pinned in ``tests/unit/gnc_control_test.cpp``
   (``Bdot.RateEnergyDecreasesMonotonicallyOverADetumbleSnippet``) and the
   duty-factor scaling in ``Bdot.DipoleScalesInverselyWithTheDutyFactor``.

.. req:: PID pointing control
   :id: REQ-ACTL-002
   :status: reviewed
   :level: L2
   :tags: adcs, control
   :method: Test
   :derived_from: REQ-MIS-001
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController
   :refs: wie1989, markley2014, astrom2008, wie2008

   The FSW **shall** provide reaction-wheel attitude control driven by the error
   rotation between the estimated attitude and a commanded reference attitude,
   with proportional, integral and derivative terms whose gains and limits are
   mission configuration.

   The error rotation **shall** take the shorter of the two rotations between the
   two attitudes. The integrator **shall** be bounded by a configured clamp and
   **shall not** accumulate while the torque command is saturated. A torque demand
   above the configured limit **shall** be scaled as a whole, preserving the
   commanded direction, and **shall not** be clipped componentwise.

   The commanded body torque **shall** be distributed across the reaction-wheel
   array so that the delivered torque equals the commanded one while every wheel
   is inside its torque limit, by either a minimum-norm or a minimum-maximum
   (L-infinity) allocation selected by configuration. Where the array cannot
   deliver the commanded torque, the per-wheel commands **shall** be scaled by a
   single factor — preserving the commanded direction — rather than clipped.

   The control mode **shall** refuse to engage, and **shall** refuse each cycle,
   unless the attitude estimate is fresh, valid, and inside a configured
   attitude-uncertainty bound.

   Holding a commanded inertial attitude under nominal sensing **with the wheel
   array near its momentum target**, the steady-state pointing error **shall**
   be below **1.0 deg**. The reference vehicle measures 0.41 deg at that
   operating point, so the bound carries a factor of 2.4 in margin; it is
   stated as a round operational number rather than shaved to the measurement.

   The momentum condition is load-bearing, not a caveat: on a **loaded** array
   the same vehicle measures **2.9°**, and the excess is arithmetically the
   wheels' Coulomb friction (the pyramid's four ``dry_friction_nm`` reactions
   sum to 2.3e-4 N·m on the body — more than twice the integrator's whole
   authority — predicting 3.0° at the committed Kp; see REQ-ACTL-010, whose
   owed friction-compensation item is what removes the condition). Push 56
   found this by measuring the requirement at an operating point Push 54 never
   visited; until the friction feedforward lands, this bound is verified only
   near zero stored momentum, and the loaded-array behaviour is bounded by
   REQ-ACTL-010's desaturation rows instead.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``InertialHoldConvergesUnderThePointingBound``), the allocation properties by
   ``tests/unit/gnc_control_test.cpp`` (``RwAllocation.*``), the anti-windup and
   short-way-round properties by ``AttitudePid.*``, and the refusal paths by the
   component test ``flight/PolarisFsw/AttitudeController/test/ut``
   (``PointRefusalPaths``, ``RefusesWithoutParameters``).

.. req:: Actuator-agnostic commanded torque
   :id: REQ-ACTL-003
   :status: reviewed
   :level: L2
   :tags: adcs, control, architecture
   :method: Inspection
   :derived_from: REQ-MIS-001
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController

   Guidance/control **shall** emit a commanded body torque, keeping upstream logic
   actuator-agnostic; the allocation layer (RW-pyramid or CMG-steering) is the
   swappable piece (see :doc:`adcs_actuators`).

.. req:: MTQ/MAG duty-cycle interlock
   :id: REQ-ACTL-004
   :status: reviewed
   :level: L2
   :tags: adcs, control, actuators, fdir
   :method: Test
   :derived_from: REQ-ADET-002, REQ-ADET-011
   :allocation: flight/PolarisFsw/AttitudeController, flight/PolarisFsw/AttitudeEstimator
   :refs: jackson1999

   An energised magnetorquer produces a near-field at a body-mounted magnetometer
   far above the ambient geomagnetic field, and a ferromagnetic rod's field
   outlives the drive. A magnetometer sample taken while a rod is energised, or
   before that field has settled, is therefore not a measurement of the
   geomagnetic field.

   Each control period **shall** be divided into an MTQ-on window and a quiet
   window, with the quiet window opening no earlier than a configured rod settle
   time after the on-window ends and the dipole commanded to zero outside the
   on-window. The duty split and the settle time **shall** be mission
   configuration, and a configuration leaving no quiet window inside the control
   period **shall** be refused.

   The schedule **shall** be published to every magnetometer consumer, and **no
   magnetometer sample whose time tag falls outside the published quiet window
   shall be consumed** — by the attitude estimators, by the magnetometer
   calibration accumulator, or by the control laws. A sample so excluded **shall**
   be reported as absent rather than as an implausible reading, so that it cannot
   latch a sensor exclusion. The count of samples excluded by the interlock
   **shall** be telemetered.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``NominalDutyCycledDetumbleDoesNotTripTheStuckOnMonitor`` — a vehicle torquing
   every cycle for a minute keeps its magnetic pair and raises no fault), the
   schedule invariants by the component test (``DutyCycleScheduleInvariants``),
   and the plant-side visibility of a violation by
   ``tests/unit/sim_io_closed_loop_test.cpp``
   (``DutyCycledRodsLeaveTheBoundaryMagnetometerSampleClean``), which shows the
   same sample corrupted by more than five times the ambient field when the rods
   are driven through the whole period.

.. req:: Stuck-on magnetorquer detection
   :id: REQ-ACTL-005
   :status: reviewed
   :level: L2
   :tags: adcs, control, actuators, fdir
   :method: Test
   :derived_from: REQ-FDIR-002
   :allocation: flight/PolarisFsw/AttitudeController

   A magnetic disturbance during the quiet window is the signature of a rod that
   failed to de-energise. The FSW **shall** monitor the measured field magnitude
   against the onboard modelled field magnitude in the quiet window and **shall**
   raise an FDIR event when the residual exceeds a configured threshold for a
   configured number of consecutive quiet windows.

   The comparison **shall** be on field *magnitudes*, which are independent of the
   attitude solution: judging a magnetometer through an attitude that magnetometer
   helped build is circular. The monitor **shall** read the magnetometer data
   **before** the plausibility band and the redundancy vote, because a disturbance
   large enough to matter is rejected as implausible by that band and a monitor
   reading the voted field would be blind to exactly the fault it exists to name.

   The event **shall** name the rod when exactly one rod carried a command in the
   period under judgement, and **shall** report the candidate set as ambiguous
   otherwise — several rods driven together are several equally good explanations
   of one residual. While the monitor is latched, the interlock **shall** be
   reported unhealthy and every magnetometer sample **shall** be excluded, since
   no window is quiet. The latch **shall** clear after a configured number of
   consecutive quiet windows passing **the same residual test that set it**, or by
   command.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``StuckOnRodIsCaughtAndTheEstimatorSurvivesIt`` — which also asserts that the
   estimator holds its fine solution through the fault it helps identify) and by
   the component test (``StuckOnMonitorLatchesAndClears``, ``StuckOnAttribution``,
   ``ResetClearsState``).

   **Owed:** resolving a multi-rod ambiguity needs a commanded isolation sweep —
   drive the rods one at a time and watch the residual — which is a recovery
   action for the Phase-7 mode manager (§10.1), not a monitor.

.. req:: Pointing-loop stability margins
   :id: REQ-ACTL-006
   :status: reviewed
   :level: L2
   :tags: adcs, control, margin
   :method: Analysis
   :derived_from: REQ-ACTL-002
   :allocation: lib/gnc, analysis/control
   :value_required: GM >= 6 dB, PM >= 30 deg
   :margin_required: 20 %
   :refs: sidi1997, astrom2008, seiler2020, franklin1998, ogata2010

   Each body axis of the reaction-wheel pointing loop, evaluated at the
   committed gains and about a linearised rigid-body attitude model, **shall**
   hold at least **6 dB of gain margin and 30 degrees of phase margin**, with a
   sensitivity peak :math:`\|S\|_\infty` no greater than 2.

   The analysis **shall** be performed on the **sampled-data** loop at the
   configured control period, including the zero-order hold, rather than on a
   continuous idealisation of it — a 10 Hz loop analysed in continuous time
   reports a phase margin the vehicle does not have.

   Because the loop is **type 3** (an integrator over a double integrator) it is
   *conditionally stable*: the phase crosses −180° below the gain crossover, at
   :math:`|L|>1`. The gain margin **shall** therefore be stated as the smaller
   of the loop-gain increase and the loop-gain decrease the loop tolerates. A
   single signed margin, which is what a naive frequency-domain margin call
   returns, reads −15 dB on this healthy design and would fail a requirement
   written for the ordinary case.

   The margins **shall** be computed from the same committed configuration the
   flight software is tuned from, never from a transcription of it, and the
   modelling assumptions the linear regime rests on — small angle, unsaturated
   torque, integrator unfrozen — **shall** be stated with the result.

   *Measured on the reference vehicle*, sampled loop at 0.1 s: X and Y hold
   **59.5°** of phase margin and **15.0 dB** of downward gain margin (37.8 dB
   upward) at a 0.272 rad/s crossover; Z holds **63.9°** and **16.6 dB**
   (36.2 dB upward) at 0.321 rad/s. Sensitivity peak is 1.02 on every axis and
   the symmetric disk margin is 0.94–1.04, i.e. 8.8–10.0 dB of *simultaneous*
   gain and 50–55° of simultaneous phase variation. The binding threshold is
   therefore cleared with a factor of 2.5 in gain margin and a factor of 2.0 in
   phase margin, and the design also clears the preferred 45° target. Adding a
   pessimistic full cycle of computation delay — which the flight topology does
   not have, since the estimator and controller run in the same 10 Hz cycle —
   costs 1.6–1.8° of phase and still passes.

   The margin extraction is Polaris's own — there is no control-systems library
   behind it — so it **shall** be validated against cases with closed-form
   answers before being applied to the vehicle. It is: the third-order loop
   :math:`1/[s(s+1)(s+2)]` whose gain margin is exactly 6 (15.563 dB) at
   :math:`\omega=\sqrt2` [ogata2010], a PD-controlled double integrator with a
   closed-form crossover and phase margin, the disk margin's exact value on a
   constant-gain loop, and the vehicle's own continuous loop against its
   hand-derived polynomial.

   Verified by ``tests/analysis/test_control_margins.py``
   (``test_committed_gains_meet_the_margin_requirement``, with the delay case
   and the conditional-stability characterisation alongside) and, through the
   shared report shape, by ``tests/analysis/test_analysis_report.py``
   (``test_control_analysis_report_passes_on_the_committed_configuration``).

   **Owed:** these are margins of the *linear*, **per-axis** loop. A saturated
   loop has no gain margin, and the describing-function or Monte Carlo
   characterisation of large-slew behaviour is Phase 11 work. The per-axis form
   assumes small stored wheel momentum, and the reference vehicle's own
   configuration exceeds that bound at wheel capacity — see the MIMO roadmap in
   design doc §8.5; the analysis emits this as a report warning rather than
   leaving it implicit.

.. req:: Actuator-suite controllability
   :id: REQ-ACTL-007
   :status: reviewed
   :level: L2
   :tags: adcs, control, actuators, redundancy
   :method: Analysis
   :derived_from: REQ-ACTL-002, REQ-ACTL-003
   :allocation: config/spacecraft, analysis/control
   :refs: wie2008, avanzini2012, astrom2008

   The reaction-wheel array **shall** render the linearised attitude model
   controllable **with any one wheel failed**, and the array geometry **shall**
   satisfy the flight allocator's own three-axis-span gate
   (``AllocMinConditioning``) in every such subset — a subset the analysis calls
   controllable but the vehicle would refuse to allocate on is not a redundancy.

   The magnetorquer-only case **shall** be documented as **instantaneously
   rank-deficient**: the torque :math:`\vec m\times\vec B` has no component
   along :math:`\hat B`, so no dipole command can rotate the vehicle about the
   local field line. Controllability from the rods alone **shall** be
   established only in the time-averaged sense, over the orbit across which the
   field direction turns.

   *Measured on the reference vehicle.* The four-wheel body-diagonal pyramid is
   controllable (Kalman rank 6 of 6) with an **isotropic** torque map:
   :math:`\lambda_{\min}/\lambda_{\max}` of :math:`AA^\top` is 1.000. All four
   3-of-4 subsets are controllable at a conditioning of **0.250**, five times
   the committed ``AllocMinConditioning`` of 0.05. The magnetorquers in a frozen
   field give rank **4 of 6** with a singular Gramian; averaged over one orbit
   of the tilted-dipole field they reach rank **6 of 6**, at a Gramian condition
   number of ~41 against ~28 for the wheels — full rank, and two orders slower
   in its weakest direction, which is why B-dot is an asymptotic law.

   Verified by ``tests/analysis/test_control_controllability.py`` and, through
   the shared report shape, by ``tests/analysis/test_analysis_report.py``.

.. req:: Attitude-estimator observability
   :id: REQ-ACTL-008
   :status: reviewed
   :level: L2
   :tags: adcs, determination, observability
   :method: Analysis
   :derived_from: REQ-ADET-002
   :allocation: lib/gnc, analysis/control
   :refs: lefferts1982, markley2014

   The attitude-error and gyro-bias states **shall** be observable from the
   vehicle's measurement configuration whenever two **non-parallel** reference
   directions are available, and the geometry metric the analysis uses **shall**
   be the same :math:`\lambda_{\min}/\lambda_{\max}` of the information matrix
   that the flight Davenport seed gate (``SeedMinObservability``) and the coarse
   chain's ``MinSinAngle`` are written on — an analysis that measured a
   different quantity from the flight gate could not be used to justify it.

   The degraded geometries **shall** be characterised rather than assumed away:
   near-parallel reference directions, where roll about the shared direction
   becomes unobservable, and **eclipse**, where the sun vector is absent
   altogether.

   *Measured on the reference vehicle* (sun total 1σ 15.7 mrad with the albedo
   correction and DE440 tables in force, uncalibrated magnetic total 33.7 mrad).
   With the two directions 90° apart the six-state model is observable, rank 6
   of 6, at an information ratio of **0.177**. At the committed ``MinSinAngle``
   gate (:math:`\sin\theta = 0.17`, i.e. 9.8°) the ratio has fallen to
   **4.2e-3**, a factor of 42, and the analysis reproduces the config's own
   closed form for it to 1e-9 — the two are the same function, not two
   implementations of one. In eclipse the model is rank **4 of 6**: rotation
   about the field direction and the bias component along it are unobservable by
   construction, which is a property of the measurement set and not a fault to
   be detected.

   Verified by ``tests/analysis/test_control_observability.py`` and, through the
   shared report shape, by ``tests/analysis/test_analysis_report.py``.

   **Owed:** this is the deterministic (rank/Gramian) statement. How fast
   information is *lost* between updates is set by the gyro random walk and
   belongs to the estimator-consistency campaign (§13), not here.

.. req:: Stored-momentum envelope
   :id: REQ-ACTL-009
   :status: reviewed
   :level: L2
   :tags: adcs, control, momentum, fdir
   :method: Test
   :derived_from: REQ-ACTL-006, REQ-ACT-003
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController, config/spacecraft
   :value_required: stored momentum <= MomentumEnvelopeNms (2.0e-3 N.m.s as committed)
   :margin_required: 10 %
   :refs: wie2008, camillo1980

   The FSW **shall** compute the wheel array's stored angular momentum
   :math:`\bar{\mathbf h} = W\,(I_w\boldsymbol\omega_w)` from the wheel
   tachometers and the configured array geometry, **shall** raise an FDIR event
   when :math:`\|\bar{\mathbf h}\|` exceeds the configured envelope, and **shall**
   refuse the computation entirely — rather than understating it — when any wheel
   reports no usable speed.

   The envelope **shall** be no larger than the stored momentum at which the
   per-axis (SISO) stability analysis behind REQ-ACTL-006 ceases to be valid,
   with at least 10 % margin. That coupling is the whole reason the envelope is
   a *small* number on this vehicle: it is not a wheel-capacity limit but the
   momentum at which the gyroscopic term :math:`\boldsymbol\omega\times(J
   \boldsymbol\omega + \bar{\mathbf h})` stops being negligible against the
   control torque at the loop crossover, past which the margins the vehicle is
   certified on describe a different system (design doc §8.5, "SISO validity
   boundary").

   *Measured on the reference vehicle.* The per-axis analysis is valid to
   **2.72e-3 N·m·s** (the binding X/Y axes, at their 0.272 rad/s crossover); the
   committed ``MomentumEnvelopeNms`` is **2.0e-3 N·m·s**, i.e. 74 % of the bound
   and inside the required margin. That is **0.4 %** of one RW-X wheel's 0.5
   N·m·s capacity — the honest statement of what the shipped linear evidence
   covers, and the reason the desaturation threshold (1.0e-3 N·m·s) sits far
   below anything a wheel would notice.

   Verified by ``tests/analysis/test_control_momentum_envelope.py``, which
   recomputes the bound from the *same committed YAML* through
   ``analysis.control.plant.siso_coupling`` rather than transcribing it, so the
   config cannot drift out of the regime its own margin evidence covers; by the
   component test (``MomentumEnvelopeAndWheelDropout``, which also pins the
   refusal on a dead tachometer and the recovery edge); and by the ``lib/gnc``
   unit tests (``MomentumManager.*``).

.. req:: Magnetic desaturation
   :id: REQ-ACTL-010
   :status: reviewed
   :level: L2
   :tags: adcs, control, momentum, actuators
   :method: Test
   :derived_from: REQ-ACT-003, REQ-ACTL-009
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController
   :refs: camillo1980, markley2014, wie2008

   The FSW **shall** unload stored wheel momentum with the magnetorquers using
   the cross-product law :math:`\mathbf m = k_d(\Delta\mathbf h\times\mathbf
   B)/\|\mathbf B\|^2`, **concurrently** with reaction-wheel pointing, engaging
   on a momentum threshold with entry/exit hysteresis and disengaging only after
   the momentum error has stayed under the exit threshold for a configured
   number of consecutive cycles.

   The demand **shall** be divided by the §7 duty factor so the average dipole
   over a control period is the commanded one, and **shall** be clamped per rod
   rather than scaled, which preserves the sign of every term of
   :math:`\mathrm{d}\|\Delta\mathbf h\|^2/\mathrm{d}t` and so keeps a saturated
   desaturation dissipative.

   Desaturation **shall not** run in DETUMBLE: the rods there are B-dot's, and
   two laws driving one actuator have no schedule that describes either. A
   ground override (``CTRL_DESAT``) **shall** be able to force or inhibit the
   decision, and forcing **shall** remain a permission — with no admissible field
   sample or no momentum estimate, no dipole is commanded.

   *Measured on the reference vehicle*, in closed loop against an unmodelled
   residual-dipole torque of ~4.5e-5 N·m: the wheels load to the 1.0e-3 N·m·s
   threshold in ~65 s, the rods engage autonomously at **1.007e-3 N·m·s**, the
   momentum falls under the 3.0e-4 N·m·s exit threshold in **~10 s** at a peak
   dipole of ~12 A·m² (of a 15 A·m² rating), and the desaturation disengages
   13 s after engaging — the dump plus the 5 s confirmation. Stored momentum
   never approaches the REQ-ACTL-009 envelope, and the cycle repeats every ~82 s
   under a disturbance that never stops.

   **Pointing during desaturation is asserted, and what it found is a vehicle
   fact rather than a control-coupling one.** The error falls through every
   desaturation window and rises while the wheels reload; on a loaded array it
   reaches 2.9°, against REQ-ACTL-002's 1.0°. The cause is the wheels' Coulomb
   friction (``dry_friction_nm`` = 1e-4 N·m; the pyramid's four reactions sum to
   2.3e-4 N·m on the body, more than twice the PID integrator's whole authority),
   not the rods: at Kp = 4.4e-3 N·m/rad that predicts 3.0° and the run measures
   2.9°. The row therefore asserts the claim this feature owns — that every
   desaturation window leaves the pointing **better** than it found it — plus an
   absolute 3.5° bound, the measurement with 20 % declared margin.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``DesaturationDumpsMomentumWhilePointingHolds`` — the full latch cycle, both
   edges, with pointing asserted against REQ-ACTL-002 throughout), by the
   component tests (``DesatEngagesAndDisengagesInPoint``,
   ``DesatExcludedFromDetumbleAndIdle``, ``DesatGroundOverride``) and by the
   ``lib/gnc`` unit tests, which pin the dissipativity argument term by term
   under the per-rod clamp (``MtqDesaturation.*``).

   **Owed (1):** torque-mode wheels on this vehicle have no **friction
   compensation**. Until the drive-level feedforward (or a speed-mode inner loop)
   lands, REQ-ACTL-002's 1° is a near-zero-momentum figure and a loaded array
   holds ~3°. That is a wheel-drive item, not a momentum-management one, and it
   is recorded here because this is the row that measured it.

   **Owed (2):** the momentum parallel to the field is untouchable at any instant —
   :math:`\mathbf m\times\mathbf B` has no component along :math:`\hat{\mathbf
   B}` — so the worst-case unloading time over the orbit's field geometry is a
   Monte Carlo campaign, as it is for B-dot (REQ-ACTL-001).

.. req:: Disturbance feedforward and the momentum-anomaly monitor
   :id: REQ-ACTL-011
   :status: reviewed
   :level: L2
   :tags: adcs, control, disturbance, fdir
   :method: Test
   :derived_from: REQ-ACTL-002, REQ-FDIR-002
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController
   :value_required: unmodelled secular torque <= DisturbanceBudgetNm (2.0e-5 N.m as committed)
   :refs: hughes1986, wertz1978, wie2008

   The FSW **shall** feed the modelled environmental torques forward into the
   pointing demand — gravity gradient :math:`3n^2\,\hat{\mathbf n}\times(J
   \hat{\mathbf n})` from the onboard attitude and position, and
   :math:`\mathbf m_{res}\times\mathbf B` from the configured residual moment
   and the onboard field (tier 1) — and **shall** estimate the *unmodelled*
   secular external torque from the rate of change of the total system momentum
   :math:`\mathbf H = J\boldsymbol\omega + \bar{\mathbf h}_w` (tier 2), feeding
   that forward as well.

   The feedforward **shall** enter the demand ahead of torque saturation and
   ahead of the integrator's anti-windup decision, and each tier **shall** be
   independently disable-able.

   The same tier-2 estimate **shall** serve as the §9 **momentum-anomaly
   monitor**: an observed unmodelled secular torque outside the configured
   budget for a configured number of consecutive updates raises an FDIR event,
   and clears below a configured clear threshold at or under the budget — held
   for the same count — with the latch holding its state between the two
   thresholds, so an estimate parked at the budget does not cycle the event
   once per confirmation count. It **shall not** fire on the modelled
   environment — in particular on the gravity-gradient torque an Earth-pointing
   vehicle sees at twice the orbital frequency, which tier 1 subtracts before the
   filter.

   The budget **shall** be derived from the observer's own noise floor at the
   committed filter length, not from the environment alone: the modelled
   disturbances on this vehicle total ~2e-7 N·m, far below what a 10 Hz momentum
   difference can resolve, so a threshold set from the physics alone would alarm
   on gyro noise. The committed 2.0e-5 N·m is ~5× the filtered floor and ~100×
   the environment.

   *Measured on the reference vehicle*, flying the **same** vehicle and the same
   injected 2.4e-4 N·m residual-dipole torque twice, differing only in whether
   the feedforward tiers are enabled: settled pointing error **3.45°** with
   feedforward against **3.52°** without, and the §9 anomaly latched **once** in
   each run — in both, because the observer runs whether or not its estimate is
   fed forward.

   **What that comparison does and does not establish.** Feedforward buys back
   the part of a disturbance the feedback loop cannot trim unaided, and at this
   operating point that part is small: the integrator absorbs up to
   Ki × clamp = 1.0e-4 N·m per axis on its own, the observer is one time constant
   into a 200 s filter, and the error budget is dominated by a term feedforward
   does not address — the wheels' Coulomb friction (REQ-ACTL-010, owed item 1),
   worth ~3.0° on a loaded array. The row therefore asserts that feedforward
   does **not degrade** the pointing and records both numbers; asserting the 2 %
   improvement itself would be a threshold inside its own noise. A decisive
   measurement of what the feedforward is worth waits on the friction term being
   removed from the budget it is compared against, and is owed.

   The anomaly monitor's own evidence is stronger and is what this row pins: it
   fires on the injected torque and, in the nominal inertial-hold row, does not
   fire at all — and it stays silent through a **full-authority detumble**, where
   B-dot's own magnetic torque is more than ten times the budget, because the
   controller subtracts the torque it commanded (the *previous* cycle's, since a
   command applies over the following interval).

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``FeedforwardImprovesPointingAndTheAnomalyMonitorFires``, a paired
   with/without comparison on one vehicle and one disturbance, plus
   ``InertialHoldConvergesUnderThePointingBound``, which is the row proving the
   monitor stays **quiet** on a nominal vehicle) and by the ``lib/gnc`` unit
   tests (``DisturbanceObserver.*``, including convergence to a known injected
   torque, the latch and its clearing, and the negative on the modelled
   gravity-gradient signature).

   **Owed:** tier 3 — fitting the residual dipole and the drag/SRP scale factors
   from long-arc data — needs the §8.3 orbit filter and is not implemented.
