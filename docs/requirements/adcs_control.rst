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

   Holding a commanded inertial attitude under nominal sensing, the steady-state
   pointing error **shall** be below **1.0 deg**. The reference vehicle measures
   0.41 deg, so the bound carries a factor of 2.4 in margin; it is stated as a
   round operational number rather than shaved to the measurement.

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
