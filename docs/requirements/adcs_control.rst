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
