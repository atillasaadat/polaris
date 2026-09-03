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

   *Both edges of the mode are one number (Push 84).* The vehicle is **tumbling**
   exactly when its body momentum :math:`|J\omega|` exceeds what the wheel array
   is certified to hold, and **detumbled** when the rods have taken it to 75 % of
   that — so ``DetumbleEnterRadps`` and ``DetumbleExitRadps`` are 100 % and 75 %
   of ``usable_momentum_nms`` divided by :math:`J_{\max}`, or **3.44 deg/s** and
   **2.58 deg/s** (0.060 and 0.045 rad/s). Both are derived and gated by
   ``analysis/sizing`` rather than tuned. The exit fraction is what makes the
   handover fast: the wheels come up already inside the regime REQ-ACTL-006's
   margins were computed for, with a quarter of the envelope left for the
   disturbance environment and the pointing transient, so Safe mode goes straight
   to coarse sun pointing instead of desaturating first.

   *Which envelope, and why not the hardware.* ``usable_momentum_nms`` is
   :math:`\min(` inscribed hardware capacity, ``MomentumEnvelopeNms`` :math:`)`
   = :math:`\min(0.049,\ 7.2\times10^{-3})` N·m·s — the certified envelope is
   15 % of what the array physically stores. Handing over at 75 % of the
   *hardware* capacity would be 0.0368 N·m·s, **3.2× outside the SISO validity
   boundary**: the wheels would hold it, REQ-ACTL-006's margins would describe a
   vehicle this is not, the §9 envelope monitor would alarm on arrival, and the
   first thing the wheels would have to do is dump momentum. 75 % is also the
   largest exit fraction the 30 % sizing-margin convention admits —
   :math:`0.75 \times 1.3 = 0.975` — so the D1b post-handover criterion passes
   with 2.6 % to spare and would fail above it.

   *The bound from below, and the margin that matters.* ``analysis/sizing``
   measures B-dot's own measurement noise floor
   :math:`\sigma\sqrt2/(\Delta t\,\|B\|)` at **1.71 deg/s** at the orbit's
   weakest field — below it the law commands on magnetometer noise rather than
   on :math:`\dot{\mathbf B}`. The 2.58 deg/s exit clears it by **51 %**. The
   previous threshold, 1.9 deg/s, cleared it by only 11 %, and that margin was
   the defect rather than a tight fit: with the predicate that close to the
   sensing floor, whether a run confirms at all is decided by field geometry
   rather than by how long the law is given, so a handover *time* written on it
   would have been a bound on luck. On the pre-Push-60 vehicle the value was
   0.5 deg/s — *below* the floor — while the momentum bound sat above it, so no
   threshold satisfied both at all.

   *The hysteresis is wide on purpose.* The 100 %/75 % pair is a 33 % band,
   against the 5 % of the old 2.00/1.90 deg/s pair. The momentum estimate carries
   observer noise, and a narrow band lets that noise walk the vehicle across the
   boundary — a mode that re-engages the rods on noise runs them continuously.
   The design closes on the tip-off it is sized for: a 5 deg/s separation is
   :math:`|J\omega| = 0.0105` N·m·s, **1.45×** the envelope, so a nominal
   tip-off does engage detumble.

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
   :refs: wie1989, wie1995, markley2014, astrom2008, wie2008

   The FSW **shall** provide reaction-wheel attitude control driven by the error
   rotation between the estimated attitude and a commanded reference attitude,
   with proportional, integral and derivative terms whose gains and limits are
   mission configuration.

   The error rotation **shall** take the shorter of the two rotations between the
   two attitudes. The integrator **shall** be bounded by a configured clamp and
   **shall not** accumulate while the torque command is saturated or the slew is
   rate-limited. A torque demand above the configured limit **shall** be scaled
   as a whole, preserving the commanded direction, and **shall not** be clipped
   componentwise. The rate the proportional term commands **shall** be limited in
   norm to a configured slew rate, so that a large error is closed by a
   constant-rate rotation about the error's eigenaxis rather than by a torque
   demand the actuators cannot carry.

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
   be below **1.0 deg**. The reference vehicle measures **0.0094 deg** at that
   operating point, so the bound carries two orders of margin; it is stated as a
   round operational number rather than shaved to the measurement, and the
   threshold has **not** been re-derived against the new measurement — tightening
   it is a requirements decision, not a consequence of a hardware change.

   The momentum condition is load-bearing, not a caveat, and it **stays** — but
   what it is guarding against has changed twice, and the second change is the
   Push 60 wheel. Push 56 found a loaded array at **2.9°** and attributed the
   excess to the wheels' Coulomb friction. Push 59 built the friction feedforward
   that item owed (REQ-ACTL-010, ``lib/gnc/rw_friction``) and then ablated the
   drive term by term instead of trusting the attribution: friction was **about
   half** of it, the loaded array measured **1.38°**, and the remainder was the
   **drive torque quantization** — the RW-X wheel that vehicle flew quotes a
   1.0e-4 N·m torque LSB, the same size as its Coulomb friction, so a per-wheel
   demand under half an LSB is commanded as zero and the loop carries a dead
   zone of ~5e-5 N·m per wheel.

   **Push 60 removed the term rather than compensating it.** ``analysis/sizing``
   found the RW-X wheel to be 78× the largest momentum any sizing driver asks of
   this 12 kg bus, and the entry was re-sized in place to the 30 mN·m·s class instead
   (design doc §19.2). That wheel's friction and drive LSB are **8e-6 N·m**, an
   order of magnitude down from 1.0e-4, and the loop was retuned around it
   (:math:`\omega_n` 0.197 → 0.8 rad/s at :math:`\zeta` = 0.7). A loaded array now
   measures **0.0396°** — 25× *inside* this requirement, where the same row
   measured 1.38° and was 1.4× outside it. The dead zone did not need a
   drive-level fix; it needed the right part.

   The near-zero-momentum condition on the **shall** above is nevertheless kept
   as written, pending review: it is the condition the requirement was verified
   under, and dropping it is a requirements change on the strength of one
   measurement of one loaded row. Loaded-array behaviour is bounded by
   REQ-ACTL-010's desaturation rows, whose absolute bound is now the rod
   contribution's own 0.05° rather than a fraction of a degree.

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
   returns, reads −27 dB on this healthy design and would fail a requirement
   written for the ordinary case.

   The margins **shall** be computed from the same committed configuration the
   flight software is tuned from, never from a transcription of it, and the
   modelling assumptions the linear regime rests on — small angle, unsaturated
   torque, integrator unfrozen — **shall** be stated with the result.

   *Measured on the reference vehicle* (the Push 60 tuning, :math:`\omega_n` =
   0.8 rad/s at :math:`\zeta` = 0.7 on the re-sized ``RW-X`` wheels), sampled loop at
   0.1 s: X and Y hold **60.1°** of phase margin at a 1.161 rad/s crossover and
   tolerate **25.5 dB** of loop-gain increase against 27.1 dB of decrease; Z
   holds **63.1°** at 1.361 rad/s and tolerates 23.9 dB up against 28.6 dB down.
   The requirement is written on the smaller of the two directions, so the gain
   margins of record are 25.5 dB (X, Y) and 23.9 dB (Z). Sensitivity peak is
   1.06–1.07 and the symmetric disk margin is 1.05–1.12, i.e. 10.1–11.0 dB of
   *simultaneous* gain and 55–59° of simultaneous phase variation. The binding
   threshold is therefore cleared with a factor of 4 in gain margin and a factor
   of 2.0 in phase margin, and the design also clears the preferred 45° target.
   Adding a pessimistic full cycle of computation delay — which the flight
   topology does not have, since the estimator and controller run in the same
   10 Hz cycle — costs **6.6–7.8°** of phase (53.5°/55.3° remaining) and still
   passes. That cost is four times the pre-Push-60 vehicle's, and for a stated
   reason rather than a mysterious one: a delay costs :math:`\omega_c T` of
   phase, and this loop's crossover is four times faster.

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
   :value_required: stored momentum <= MomentumEnvelopeNms (7.2e-3 N.m.s as committed)
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
   **1.16e-2 N·m·s** (the binding X/Y axes, at their 1.161 rad/s crossover); the
   committed ``MomentumEnvelopeNms`` is **7.2e-3 N·m·s**, i.e. 62 % of the bound
   and inside the required margin. That is **24 %** of one wheel's
   0.030 N·m·s capacity and 15 % of the 0.049 N·m·s the four-wheel pyramid
   guarantees in every direction — a certified regime that is now a real
   fraction of the hardware rather than the 0.4 % of a wheel the pre-Push-60
   vehicle's envelope covered. The bound scales as :math:`J_{\min}\omega_c`, so
   what closed that gap is the loop bandwidth (0.197 → 0.8 rad/s), which the
   30 mN·m·s wheel is what made affordable; the desaturation threshold
   (3.6e-3 N·m·s) sits at half the envelope.

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
   residual-dipole torque of ~4.5e-5 N·m: the wheels load to the committed
   3.6e-3 N·m·s entry threshold, the rods engage autonomously there, the
   momentum falls under the 1.08e-3 N·m·s exit threshold and the latch clears
   after its 5 s confirmation, settling at 6.2e-4 N·m·s. Peak stored momentum
   over the run is **3.60e-3 N·m·s**, half of the 7.2e-3 N·m·s REQ-ACTL-009
   envelope, under a disturbance that never stops. (The dump and reload
   *durations* moved with the Push 60 thresholds and are not restated here until
   the row records them; what the row asserts is the momentum, both latch edges
   and the pointing below.)

   **Pointing during desaturation is asserted, and what it found is a vehicle
   fact rather than a control-coupling one.** On a loaded array the worst error
   is **0.0396°** against REQ-ACTL-002's 1.0° — 25× inside the requirement,
   where the pre-Push-60 vehicle measured 1.38° and was outside it. The row
   asserts an absolute **0.05°** bound on the error while a rod is energised,
   which is where the requirement is written, plus a loose 5× band against the
   quiet phases as a sanity check.

   **The relative claim was retired in Push 60 and should not be restored.**
   Through Push 59 this row asserted that the worst pointing while a rod is
   energised is no worse than the worst while none is (0.92° against 1.38°), an
   inequality that meant something while the wheels' own Coulomb friction
   dominated the budget and the rods were a small perturbation on a large
   number. The re-sized wheel carries 8e-6 N·m of friction rather than 1e-4, and
   with the Push 59 feedforward on top the quiet error collapses to **0.0203°**
   — at which point the rods are the *dominant* term inside their own windows
   and the ratio rises to ~2 while the absolute error stays 25× inside the
   requirement. A ratio against a near-zero baseline is evidence that everything
   else got quiet, not evidence of coupling, so the rod contribution is bounded
   absolutely and the ratio kept only as a band.

   Push 56's earlier version of the same paragraph asserted that every
   desaturation window left the pointing **better** than it found it, on the
   mechanism that emptying the wheels removed their friction torque. Push 59's
   friction feedforward paid for that friction directly, so that coupling — and
   with it that claim — was already gone. Both retirements are the same
   discipline: the assertion moves with the physics rather than being kept as
   something the vehicle no longer supports.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``DesaturationDumpsMomentumWhilePointingHolds`` — the full latch cycle, both
   edges, with pointing asserted against REQ-ACTL-002 throughout), by the
   component tests (``DesatEngagesAndDisengagesInPoint``,
   ``DesatExcludedFromDetumbleAndIdle``, ``DesatGroundOverride``) and by the
   ``lib/gnc`` unit tests, which pin the dissipativity argument term by term
   under the per-rod clamp (``MtqDesaturation.*``).

   **Owed (1) — implemented in Push 59.** The drive-level **friction
   feedforward** now runs: ``lib/gnc/rw_friction`` adds :math:`-\tau_f` per wheel
   after the §8.5 allocation and before the drive command, so the net rotor
   torque is the one the allocation asked for. The Coulomb term's
   :math:`\mathrm{sgn}(\omega)` goes through a bounded linear blend over a
   configured deadband speed (Karnopp's zero-velocity band in feedforward form
   [karnopp1985]) rather than a bare sign, which would chatter at a wheel's zero
   crossing; inside the band the friction is deliberately under-compensated, and
   that is stated as the band's cost rather than hidden. The compensation never
   exceeds the modelled friction at a trim :math:`k \le 1`, and the torque-box
   clamp is applied so that what it removes is the compensation and never the
   control demand. A wheel with no usable tachometer has no speed sign and is
   passed through uncompensated rather than given a guessed one.

   *Measured on the pre-Push-60 reference vehicle* (the RW-X wheels, whose
   1.0e-4 N·m Coulomb friction is what the feedforward was built against), worst
   pointing over the same orbit and seed: **2.77° uncompensated → 1.38°** at the
   flown trim (``WheelFrictionScale`` = 0.5). Those figures are the evidence for
   the *law*, and they stand as taken; on the re-sized wheels the friction being
   compensated is an order of magnitude smaller, so the same feedforward is worth
   correspondingly less and the loaded array measures 0.0396° with it flown. The
   trim is a genuine closed-loop parameter
   and not merely a safety margin on a model: full compensation is *worse* than
   half on both operating points (1.52° loaded and 1.09° near-empty at
   :math:`k = 1.0`, against 1.38° and 0.25° at :math:`k = 0.5`), because the
   friction being removed was also passively damping the stored wheel momentum
   and a near-empty array runs at the speeds where the compensation's sign is
   least trustworthy. The open-loop argument that partial compensation is never
   worse is about the *residual disturbance*, and it does not carry to the closed
   loop; the numbers do, and they are why the trim is a flight parameter.

   Verified by the ``lib/gnc`` unit tests (``RwFriction.*`` — blend continuity
   and boundedness, the no-over-compensation bound, saturation, the trim, and a
   zero-speed no-chatter case), by the component test
   ``WheelFrictionFeedforward`` (the commanded torque is the allocation's demand
   plus the modelled friction and the two are separable from telemetry; a missing
   coefficient refuses the whole configuration rather than flying the feedforward
   silently off), and by the SITL row below.

   **Owed (2) — the wheel drive's torque quantization, and what Push 60 did to
   it.** The Push 59 ablation named a 1.0e-4 N·m drive LSB — a ±5e-5 N·m
   per-wheel dead zone — as ~1.2° of the remaining 1.38°, and recorded it as the
   term standing between a loaded array and REQ-ACTL-002's 1.0°. **It is no
   longer that term.** The wheel as re-sized in Push 60 quotes an
   8e-6 N·m LSB, an order of magnitude finer, and the loaded array now measures
   0.0396° — inside the requirement with 25× to spare, without any of the
   drive-level fixes (dither, a speed-mode inner loop, a finer drive) this item
   proposed. The item stays open only as a *class*: a dead zone of half an LSB
   is still the pointing floor a wheel drive imposes, and it is what would bind
   again on a vehicle whose drive is coarse relative to its control torque. It
   no longer names a gap against a requirement on this vehicle.

   **Owed (3) — the blend deadband's lower bound.** The flown 5.0e-3 rad/s is
   derived from this vehicle's own wheel-speed range, which the Push 60 wheel
   moved: the 30 mN·m·s rotor's inertia is 4.775e-5 kg·m², so the committed
   1.08e-3 to 3.6e-3 N·m·s momentum band puts each wheel between roughly **10
   and 33 rad/s** (~90 to 310 rpm) rather than the ~0.05 to 0.55 rad/s the
   ten-rpm RW-X vehicle ran at. The flown band is therefore now ~0.02 % of the
   upper end rather than 1 % of it — further inside the "small against the
   operating range" rule the value was chosen under, not outside it. The *lower* bound is the flight tachometer's resolution and noise,
   below which the sign of :math:`\omega` is not a measurement — and the re-sized
   catalog entry, like the ``RW-X`` one before it, carries no tachometer
   specification, so that bound is unknown for this vehicle and must be
   re-derived against the flight unit before launch.

   **Owed (4):** the momentum parallel to the field is untouchable at any instant —
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
   the feedforward tiers are enabled: settled pointing error **0.0409°** with
   feedforward against **0.0391°** without, and the §9 anomaly latched **once**
   in each run — in both, because the observer runs whether or not its estimate
   is fed forward.

   **What that comparison establishes, and how it has moved.** Push 56 measured
   the same pair at 3.45° against 3.52° — a 2 % difference, too small to assert.
   Push 59's friction feedforward (REQ-ACTL-010, owed item 1) took the wheels'
   Coulomb friction out of the budget the comparison runs against, and the pair
   separated to 1.72° against 2.10°, an 18 % effect that was the first decisive
   reading of what the tiers are worth. Push 60's wheel and its retuned loop then
   shrank the *whole* budget by a further factor of ~40, and at 0.04° the pair is
   **inside its own noise again, with the feedforward run 4.6 % worse rather than
   better**. That is not a regression in the feedforward: the disturbance it
   subtracts is unchanged, and what changed is that a 16× stiffer proportional
   gain now trims the same torque almost entirely on its own, leaving a
   difference of ~2e-4° between the two runs. The row's assertion is unchanged
   and is the one that survives all three readings — feedforward **does not
   degrade** the pointing (asserted with a 5 % band), with both numbers recorded.
   **Asserting an improvement is off the table on this vehicle**, not deferred:
   the effect is smaller than the measurement, and a vehicle whose PID alone
   holds 0.04° is not the vehicle on which feedforward earns its keep.

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

   **Tier 3 status.** Both halves are now implemented and both measured that
   they cannot resolve their parameter on *this* vehicle, which is the honest
   result rather than a gap: the residual-dipole fit (``gnc::DipoleEstimator``,
   Push 59) sits ~40x under the tier-2 observer's noise floor, and the **drag
   scale factor** (``gnc::OrbitOd``, Push 76, REQ-ODP-013) ~39x under the
   process noise that covers the 8x8 geopotential truncation. Each ships tested,
   parameterised and disabled, publishing the sigma that makes its own limit
   legible. Still owed: an **SRP** scale factor, which needs an onboard SRP term
   before it can have a coefficient to scale.

.. req:: Reaction-wheel speed bias and per-wheel capacity monitoring
   :id: REQ-ACTL-012
   :status: reviewed
   :level: L2
   :tags: adcs, control, momentum, fdir
   :method: Test
   :derived_from: REQ-ACTL-002, REQ-ACTL-009
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeController, sim/actuators
   :value_required: no wheel zero crossing after the bias servo has converged; every wheel < WheelCapacityNms
   :refs: markley2014, karnopp1985

   The FSW **shall** be able to hold every reaction wheel at a configured
   non-zero speed while pointing, so that the wheels operate away from zero
   speed — where the drive's torque resolution, the Coulomb friction sign
   reversal and stiction degrade the delivered torque — and it **shall** do so
   without storing body momentum on a redundant array: the bias **shall** live
   in the null space of the wheel torque map, its command **shall** produce
   zero body torque for any configured pattern (only the null-space projection
   of the pattern being pursued), and it **shall** be a slow trim bounded by a
   configured torque cap applied as one scale on the whole command. The bias
   **shall** be switchable off by configuration (an all-zero pattern) and
   **shall** be inert on an array with no null space. On such an array the
   bus-level momentum target of REQ-ACTL-009 is the alternative.

   The FSW **shall** monitor the largest single-wheel momentum against the
   wheel's capacity and raise an edge-gated event above a configured fraction
   of it, because a redundant array can hold null-space momentum that no
   body-momentum threshold sees; the null-space content **shall** be
   telemetered.

   The truth simulator **shall** model the physics the bias exists for: the
   torque box and speed ceiling with the delivered torque backed out of the
   clamped motion, drive quantisation, the Coulomb sign reversal through zero,
   and Karnopp stiction holding a rotor inside a configured band against a
   sub-breakaway command.

   Rationale: on the reference vehicle a zero-momentum hold under a small
   secular load crossed wheel zero speed 21 times in 250 s and pointed to
   0.0144°; with the null-space bias at 10 % of capacity there were no
   crossings, the slowest wheel stayed out of the stiction band, and the hold
   pointed to 0.0119° (Push 69). Sizing rule, enforced by the config compiler:
   the bias must be under the wheel's capacity and at least 0.75× the
   desaturation entry threshold (the worst single-wheel share of it on the
   pyramid), or a loaded wheel reaches zero before desaturation engages.

   Verified by ``tests/integration/sitl_attitude_control_test.cpp``
   (``WheelSpeedBiasKeepsTheWheelsOffZero``, the same loaded hold flown with
   and without the bias), the ``AttitudeController`` component tests
   (``WheelBiasServoAddsNullSpaceTorqueOnly``, ``WheelBiasPastCapacityIsRefused``,
   ``WheelCapacityMonitorSeesNullSpaceMomentum``), ``lib/gnc`` unit tests
   (``RwBias.*``: zero body torque over random patterns and momenta with the cap
   engaged, convergence to the pattern, the projection of a mis-set pattern,
   inertness on three wheels), the sim wheel's stiction tests
   (``ReactionWheel.Stiction*``), and the config compiler's bias rules
   (``tests/tools/test_config_compiler.py``).
