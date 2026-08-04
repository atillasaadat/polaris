module flight {

  # ----------------------------------------------------------------------
  # Attitude control (design doc §8.5, §7, §9, §10)
  # ----------------------------------------------------------------------
  #
  # The vehicle's torque authority. It runs on the barrier-driven 10 Hz GNC rate
  # group (§2.4) as the member **after** the estimator, consumes the §8.0
  # attitude estimate that member published this same cycle, and commands the
  # actuators for the next interval:
  #
  #   DETUMBLE -> B-dot dipole on the magnetorquers (lib/gnc/bdot.hpp)
  #   POINT    -> quaternion-error PID (lib/gnc/attitude_pid.hpp) allocated
  #               across the reaction-wheel array (lib/gnc/rw_allocation.hpp)
  #   IDLE     -> zero on both
  #
  # No control math lives here. The laws are in the flight-safe `lib/gnc`, so the
  # component is the wiring, the mode ladder, the §7 duty-cycle schedule and the
  # §9 monitors — which is exactly the split the AttitudeEstimator follows.
  #
  # **This component owns the MTQ/MAG duty-cycle interlock** (§7). It is the only
  # thing on the vehicle that energises a torque rod, so it is the only thing that
  # can say when the field is quiet: each control period splits into an MTQ-on
  # window and a quiet window opening one rod settle-time later, the dipole is
  # commanded to zero outside the on-window, and the schedule is published on
  # `mtqActuationOut` for the estimator to gate its magnetometer samples with. The
  # matching stuck-on monitor lives here too, because the comparison it needs —
  # "the field is disturbed and I commanded nothing" — is only available to the
  # thing doing the commanding.
  #
  # **Passive, and the same threading argument as the estimator.** Commands
  # arrive on the command dispatcher's thread while `run` executes on the rate
  # group's, so every piece of state both touch is behind `guarded` ports: the
  # mode, the target quaternion, the reset. The estimate input is `guarded` too:
  # it is written on the rate group's thread but *read* by the command handlers
  # (`CTRL_MODE_SET`'s quality floor), and guarding only one side of that pair is
  # no mutual exclusion at all — a torn `estimate_` is finite, plausible, and
  # wrong, deciding a mode entry. There is no re-entry risk: the estimator
  # invokes the port from outside any of this component's handlers.
  #
  # **Autonomous entry into DETUMBLE on rate is deliberately not here.** §10 makes
  # that the mode manager's decision, and the mode manager is Phase 7. This push
  # ships the *predicate* (the rate hysteresis, telemetered as DetumbleComplete)
  # and commanded transitions only, so the state machine has something to read
  # when it arrives rather than a second opinion to disagree with.
  passive component AttitudeController {

    # ----------------------------------------------------------------------
    # Rate group
    # ----------------------------------------------------------------------

    @ Control cycle entry, at the GNC rate (10 Hz, §2.4). Runs after the
    @ estimator on the same cycle, so it acts on this step's solution.
    guarded input port run: Svc.Sched

    # ----------------------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------------------

    @ The §8.0 attitude estimate, from the AttitudeEstimator. Carries the
    @ attitude, the body rate, the covariance diagonal the quality floor is
    @ applied to, **and** the voted, calibrated, interlock-gated magnetic field
    @ the B-dot law differences. Reading the magnetometer port array directly
    @ instead would be a second opinion about which unit the vehicle believes.
    guarded input port estimateIn: AttitudeEstimatePort

    # ----------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------

    @ Per-wheel torque commands [N*m] in vehicle build order (§7).
    output port wheelCmdOut: WheelTorqueCmd

    @ Per-rod dipole commands [A*m^2] plus this period's MTQ-on window length —
    @ the plant needs the window to apply the rods over the right fraction of the
    @ step, and to corrupt any magnetometer sample that lands inside it.
    output port mtqCmdOut: MtqDipoleCmd

    @ This period's duty-cycle schedule and interlock verdict, to every
    @ magnetometer consumer (§7 layer 2).
    output port mtqActuationOut: MtqActuationPort

    # ----------------------------------------------------------------------
    # Types
    # ----------------------------------------------------------------------

    @ Control mode (design doc §8.5). Not the mission mode (§10) — that is the
    @ Phase-7 state machine's, and it will command this one.
    enum CtrlMode : U8 {
      IDLE = 0 @< zero on every actuator; the mode a refusal falls back to
      DETUMBLE = 1 @< B-dot rate reduction on the magnetorquers
      POINT = 2 @< quaternion-error PID on the reaction wheels
    }

    @ Which allocation the wheel array is driven with (§8.5).
    enum AllocMethod : U8 {
      MIN_NORM = 0 @< L2 pseudo-inverse: least total wheel torque
      MIN_MAX = 1 @< L-infinity: least largest wheel torque (four wheels or fewer)
    }

    @ Why a commanded mode change or a control cycle was refused.
    enum CtrlRefusal : U8 {
      NOT_CONFIGURED = 0 @< a required parameter is missing or out of range
      NO_ESTIMATE = 1 @< no estimate has arrived, or it is stale
      ATTITUDE_INVALID = 2 @< the estimate reports no usable attitude
      RATE_INVALID = 3 @< the estimate reports no usable body rate
      QUALITY_FLOOR = 4 @< the attitude uncertainty is above MaxAttSigmaRad
      NO_TARGET = 5 @< POINT with no target quaternion ever commanded
      NO_FIELD = 6 @< DETUMBLE with no admissible magnetometer sample
      ALLOCATION = 7 @< the wheel allocation refused this cycle
      BAD_COMMAND = 9 @< a command was formed but is not finite — a numerics fault rather than a sensing one, so it is reported apart from NO_FIELD
      FIELD_INTERVAL = 8 @< a field sample arrived but no derivative could be formed from it (first sample of a pair, or a spacing outside the configured band); distinct from NO_FIELD because the causes and the fixes differ
    }

    @ Why the magnetorquer stuck-on monitor could not name a rod.
    enum StuckAttribution : U8 {
      DECISIVE = 0 @< exactly one rod carried a command; the EVR names it
      AMBIGUOUS = 1 @< several rods (or none) were commanded; only the mask is known
    }

    # ----------------------------------------------------------------------
    # Commands
    # ----------------------------------------------------------------------

    @ Set the control mode. POINT is refused — with a ModeRefused event, leaving
    @ the current mode in place — unless the estimate meets the configured
    @ quality floor and a target has been set; DETUMBLE is refused without an
    @ admissible field. IDLE is always accepted, because the way out of a bad
    @ state must never itself have a precondition.
    guarded command CTRL_MODE_SET(
                                   $mode: CtrlMode @< requested control mode
                                 ) \
      opcode 0

    @ Set the inertial-hold reference attitude (Body <- ECI, JPL scalar-first).
    @ Normalised on receipt and made canonical (q0 >= 0); a non-finite or
    @ null-norm quaternion is rejected and the previous target stands. Full
    @ guidance modes — nadir, LVLH, sun-point, ground-track — are §8.4/Phase 7;
    @ this is the one reference a controller can carry without a guidance layer.
    guarded command CTRL_SET_TARGET_Q(
                                       q0: F64 @< scalar part
                                       q1: F64
                                       q2: F64
                                       q3: F64
                                     ) \
      opcode 1

    @ Drop the controller's accumulated state: the PID integrator, the B-dot
    @ previous sample, the rate-hysteresis streak, the stuck-on latch and its
    @ counters; re-read the tuning; fall back to IDLE. The commanded
    @ re-admission path for a rod latched out by the stuck-on monitor.
    guarded command CTRL_RESET \
      opcode 2

    # ----------------------------------------------------------------------
    # Parameters (design doc §19.3 — no defaults; a missing value refuses)
    # ----------------------------------------------------------------------

    @ Control period [s]. The rate group's own period, and the span the duty-cycle
    @ schedule divides. Declared rather than measured because the schedule must be
    @ deterministic: deriving it from the last two cycle epochs would make one
    @ late cycle silently move the quiet window a magnetometer sample is judged
    @ against.
    param ControlPeriodSec: F64

    @ Largest age [s] of an attitude estimate the controller will act on. Older
    @ than this and the cycle is refused: acting on a stale attitude is how a
    @ controller drives a vehicle to where it used to be.
    param MaxEstimateAgeSec: F64

    @ Quality floor for POINT [rad]: the largest per-axis attitude 1-sigma
    @ (sqrt of the largest covariance diagonal) the mode will engage or stay on.
    @ Mode-agnostic on purpose — what a pointing law needs is a bound on the
    @ error it is closing, not a label saying which estimator produced it.
    param MaxAttSigmaRad: F64

    # --- B-dot detumble (lib/gnc/bdot.hpp) ---------------------------------

    @ B-dot gain k [N*m*s] in m = -(k/||B||^2) dB/dt. Floor is Avanzini &
    @ Giulietti's 2*omega_orbit*(1+sin xi)*J_min; above it the closed-loop rate
    @ time constant is J/(k*duty) until the rods saturate.
    param BdotGainNms: F64

    @ Per-rod rated magnetic moment [A*m^2]. Each dipole component is clamped to
    @ it independently, which preserves the sign of every term in the energy rate
    @ and so keeps a saturated B-dot dissipative.
    param BdotMaxDipoleAm2: F64

    @ Shortest usable spacing between two field samples [s] for the derivative.
    param BdotMinSampleDtSec: F64

    @ Longest usable spacing [s]. Past it the stored sample is dropped and the law
    @ re-acquires from the next pair rather than differencing across a gap.
    param BdotMaxSampleDtSec: F64

    @ Body-rate magnitude [rad/s] above which the vehicle counts as tumbling.
    param DetumbleEnterRadps: F64

    @ Body-rate magnitude [rad/s] below which, held for DetumbleConfirmCycles,
    @ detumble is complete. Must be below DetumbleEnterRadps — the deadband is
    @ what stops the verdict chattering.
    param DetumbleExitRadps: F64

    @ Consecutive cycles under DetumbleExitRadps that confirm completion.
    param DetumbleConfirmCycles: U32

    # --- Pointing PID (lib/gnc/attitude_pid.hpp) ----------------------------

    @ Proportional gain on the error rotation vector [N*m/rad].
    param PidKpNmPerRad: F64

    @ Integral gain [N*m/(rad*s)]. Zero flies a PD controller.
    param PidKiNmPerRadS: F64

    @ Derivative gain on the rate error [N*m/(rad/s)].
    param PidKdNmPerRadps: F64

    @ Per-axis integrator clamp [rad*s].
    param PidMaxIntegralRadS: F64

    @ Magnitude limit on the commanded body torque [N*m] — the vehicle's usable
    @ three-axis authority, before the per-wheel limits of the allocation.
    param PidMaxTorqueNm: F64

    @ Longest step [s] the integrator is advanced over. A larger gap is a dropout:
    @ the PD terms still act, the integral is not advanced across it.
    param PidMaxDtSec: F64

    # --- Wheel allocation (lib/gnc/rw_allocation.hpp) -----------------------

    @ Number of installed reaction wheels, 3..GncMaxUnits.
    param WheelCount: U32

    @ Each wheel's **spin axis** in body frame, flattened three at a time in
    @ vehicle build order — the same numbers the vehicle config gives the sim's W
    @ matrix. The controller negates them to get the torque-authority axes, since
    @ a wheel's reaction on the body is -I*omega_dot; doing that here, once, is why
    @ the config carries the physical axis rather than a pre-signed one.
    param WheelAxesBody: Vec3F64PerUnit

    @ Per-wheel commanded-torque limit [N*m]. One value: the reference vehicle
    @ flies four identical wheels, and a mixed array needs a per-unit array here
    @ (and in the allocator's config, which already takes one).
    param WheelMaxTorqueNm: F64

    @ Smallest acceptable lambda_min/lambda_max of A*A^T — the three-axis-span
    @ gate on the wheel array. Scale-free, so it gates geometry rather than wheel
    @ sizing.
    param AllocMinConditioning: F64

    @ Which allocation to run (§8.5), mirroring `AllocMethod`: 0 = MIN_NORM,
    @ 1 = MIN_MAX. A U8 rather than the enum because the config compiler's
    @ ParameterDb emitter encodes F´ scalar and array types only, and an enum
    @ parameter would fail there rather than in flight — an honest narrowing, not
    @ a modelling choice. Any other value is a ConfigInvalid. MIN_MAX falls back
    @ to MIN_NORM with one AllocationFallback event on an array whose null space
    @ is larger than one dimension, rather than pretending.
    param AllocMethodSel: U8

    # --- MTQ/MAG duty-cycle interlock (§7) ---------------------------------

    @ Number of installed magnetorquer rods. This push drives an orthogonal triad
    @ and refuses anything else, so it must be 3.
    param MtqCount: U32

    @ Each rod's dipole axis in body frame, flattened three at a time in vehicle
    @ build order. The commanded body dipole is resolved onto them, clamped per
    @ rod, and re-expanded — which is only correct for a mutually orthogonal set,
    @ so a non-orthogonal one is refused with ConfigInvalid rather than
    @ approximated.
    param MtqAxesBody: Vec3F64PerUnit

    @ On-window fraction of the control period, in (0, 1]. Trades magnetic
    @ authority (the average dipole scales with it, which B-dot compensates for)
    @ against how much of the period is left for a clean magnetometer sample.
    param MtqDutyFactor: F64

    @ Rod settle time [s]: the L/R current decay plus core relaxation after the
    @ drive is removed, from the rod's hardware-catalog entry. The quiet window
    @ opens this long after the on-window ends, and nothing before it is a
    @ measurement of the geomagnetic field.
    param MtqSettleSec: F64

    @ Late-sample tolerance on the quiet window's **end** [s]. The rods are off
    @ from the on-window's end until the next period's on-window begins, so a
    @ sample arriving a little after the nominal boundary is still quiet;
    @ extending the window's *start* would do the opposite and admit a dirty one.
    @ Sized for scheduler jitter and time-tag rounding, not for a scheduling
    @ failure — a period that is wrong rather than jittery is reported by
    @ CyclePeriodMismatch, not absorbed here.
    param MtqWindowToleranceSec: F64

    @ Field-magnitude residual [T] against the onboard IGRF that counts as a
    @ disturbance during a quiet window (§9). The comparison is on **magnitudes**,
    @ which is attitude-free: judging a magnetometer through an attitude the
    @ magnetometer helped build is the circularity that latches out healthy units.
    param MtqStuckResidualT: F64

    @ Consecutive quiet windows over MtqStuckResidualT that latch a stuck-on rod.
    param MtqStuckConfirmCycles: U32

    @ Consecutive quiet windows back under MtqStuckResidualT that clear the latch.
    @ The same criterion that set it, which is what stops an exclusion becoming a
    @ life sentence.
    param MtqStuckClearCycles: U32

    @ Cadence [cycles] for repeatable warnings (saturation, refusals): one event,
    @ then one per this many cycles while the condition persists. A 10 Hz loop
    @ that events every cycle floods the downlink and hides the transition.
    param AlertCycles: U32

    # ----------------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------------

    @ Active control mode.
    telemetry CtrlModeTlm: CtrlMode

    @ Commanded body torque [N*m] before allocation (POINT only; zero otherwise).
    telemetry TorqueCmd: Vec3F64

    @ Commanded body dipole [A*m^2] during the on-window (DETUMBLE only).
    telemetry DipoleCmd: Vec3F64

    @ Per-wheel commanded torque [N*m], in vehicle build order.
    telemetry WheelTorque: F64PerUnit

    @ Largest |wheel torque| in this cycle's allocation [N*m] — the quantity the
    @ L-infinity allocation minimises, so the two methods are comparable in flight.
    telemetry MaxWheelTorque: F64

    @ Fraction of the commanded body torque the wheels actually delivered, in
    @ (0, 1]. Below 1 means the array saturated and the torque was scaled, not
    @ clipped: same direction, less of it.
    telemetry AllocScale: F64

    @ Estimated body-rate magnitude [rad/s] this cycle.
    telemetry RateNorm: F64

    @ Pointing error [rad] against the commanded target — the atan2 form, exact
    @ near zero, which is where a pointing controller lives.
    telemetry PointingErrorRad: F64

    @ The rate hysteresis says detumble is complete (rate held under
    @ DetumbleExitRadps for DetumbleConfirmCycles). The predicate the Phase-7 mode
    @ manager will read; nothing acts on it yet.
    telemetry DetumbleComplete: bool

    @ No rod is latched stuck-on. False makes every magnetometer sample suspect
    @ downstream, which is why it is telemetered next to the mode.
    telemetry MtqInterlockHealthy: bool

    @ Bit i set means rod i is latched stuck-on by the §9 monitor.
    telemetry MtqStuckMask: U32

    @ |measured - modelled| field magnitude [T] in this cycle's quiet window, the
    @ quantity MtqStuckResidualT gates. No value when there was no admissible
    @ sample or no modelled field.
    telemetry MagResidualT: F64

    @ Control cycles refused since start (any CtrlRefusal). A climbing count with
    @ the mode stuck at IDLE is the signature of a controller that never engaged.
    telemetry CyclesRefused: U32

    @ Control cycles executed since start.
    telemetry CyclesRun: U32

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ The control mode changed (commanded, or fallen back to IDLE on a refusal).
    event ModeChanged(from: CtrlMode, to: CtrlMode) \
      severity activity high \
      format "Control mode {} -> {}"

    @ A commanded mode change was refused; the previous mode stands.
    event ModeRefused(requested: CtrlMode, reason: CtrlRefusal) \
      severity warning low \
      format "Control mode {} refused: {}"

    @ The active mode could not run this cycle. Edge-gated and then repeated at
    @ the AlertCycles cadence, so a persistent cause costs a bounded event stream.
    event ControlRefused(activeMode: CtrlMode, reason: CtrlRefusal, cycles: U32) \
      severity warning low \
      format "Control refused in mode {}: {} ({} cycles)"

    @ A required parameter is missing or out of range; the controller is inert in
    @ IDLE and commands zero. Edge-gated.
    event ConfigInvalid(detail: string size 80) \
      severity warning high \
      format "AttitudeController configuration invalid: {}"

    @ The commanded body torque hit PidMaxTorqueNm, or the wheel array saturated.
    @ Bounded cadence.
    event TorqueSaturated(demandNm: F64, limitNm: F64, allocScale: F64) \
      severity warning low \
      format "Torque saturated: demand {} N*m against limit {} N*m, allocation scale {}"

    @ The commanded dipole hit the per-rod limit. Bounded cadence. Normal at the
    @ start of a detumble from a high rate, which is why it is a low warning.
    event DipoleSaturated(dipoleAm2: F64, limitAm2: F64) \
      severity warning low \
      format "Dipole saturated: {} A*m^2 against per-rod limit {} A*m^2"

    @ L-infinity allocation was configured but the array's null space is larger
    @ than one dimension; the cycle used L2 instead. Emitted once per
    @ configuration, not per cycle.
    event AllocationFallback(wheels: U32) \
      severity warning low \
      format "Min-max allocation unavailable for a {}-wheel array; using min-norm"

    @ A field-magnitude disturbance persisted through the quiet window with the
    @ rods de-energised: a stuck-on rod (§7 layer 3, §9.2). `unit` names the rod
    @ when exactly one carried a command this period, and is 255 otherwise —
    @ three rods driven together are three equally good explanations, and naming
    @ one of them would be a guess. The `candidateMask` is what is actually known.
    @ Resolving a multi-rod ambiguity needs a commanded isolation sweep, which is
    @ the Phase-7 state machine's recovery action.
    event MtqStuckOn(unit: U8, attribution: StuckAttribution, candidateMask: U32, \
                     residualT: F64, cycles: U32) \
      severity warning high \
      format "Magnetorquer stuck-on: rod {} ({}), candidates 0x{x}, residual {} T after {} quiet windows"

    @ The disturbance cleared for MtqStuckClearCycles consecutive quiet windows —
    @ the same criterion that latched it. The recovery edge that keeps the
    @ exclusion from being permanent.
    event MtqStuckCleared(mask: U32, cycles: U32) \
      severity activity high \
      format "Magnetorquer stuck-on cleared (was 0x{x}) after {} clean quiet windows"

    @ The measured spacing between control cycles disagrees with
    @ ControlPeriodSec by more than MtqWindowToleranceSec. The §7 quiet window is
    @ computed from the *declared* period, so a real period that differs from it
    @ puts the window in the wrong place and every magnetometer sample is
    @ silently rejected — which on a vehicle looks like the magnetometers
    @ failing. Bounded cadence.
    event CyclePeriodMismatch(measuredSec: F64, declaredSec: F64, cycles: U32) \
      severity warning high \
      format "Control cycle period {} s against the declared {} s ({} cycles): the §7 quiet window is misplaced"

    @ A new inertial-hold target was accepted.
    event TargetSet(q0: F64, q1: F64, q2: F64, q3: F64) \
      severity activity high \
      format "Inertial hold target set to [{}, {}, {}, {}]"

    @ A CTRL_SET_TARGET_Q was rejected; the previous target stands.
    event TargetRejected \
      severity warning low \
      format "Inertial hold target rejected: non-finite or null-norm quaternion"

    @ CTRL_RESET executed.
    event ControllerReset \
      severity activity high \
      format "Attitude controller reset: integrator, B-dot history and stuck-on latch cleared"

    # ----------------------------------------------------------------------
    # Standard AC ports
    # ----------------------------------------------------------------------

    @ Port for requesting the current time (TAI under SITL, via SitlTime).
    time get port timeCaller

    @ Parameter get port
    param get port prmGetOut

    @ Parameter set port
    param set port prmSetOut

    @ Command registration port
    command reg port cmdRegOut

    @ Command receive port
    command recv port cmdIn

    @ Command response port
    command resp port cmdResponseOut

    @ Event port
    event port logOut

    @ Text event port
    text event port logTextOut

    @ Telemetry port
    telemetry port tlmOut

  }

}
