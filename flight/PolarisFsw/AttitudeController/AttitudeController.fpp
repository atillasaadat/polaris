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
  #               across the reaction-wheel array (lib/gnc/rw_allocation.hpp),
  #               **plus**, concurrently, cross-product desaturation on the rods
  #               when the wheels are loaded (lib/gnc/momentum.hpp)
  #   IDLE     -> zero on both
  #
  # **Desaturation is concurrent with POINT and excluded from DETUMBLE**, which
  # is the one piece of mode logic worth stating up front: the wheels hold the
  # attitude while the rods dump the momentum they are holding, so the two laws
  # run in the same cycle on different actuators — but in DETUMBLE the rods are
  # B-dot's, and two laws driving one actuator would be two vehicles' worth of
  # commands on one set of coils. The seam is drawn where the dipole is *formed*:
  # exactly one law produces a body dipole in any cycle, and `commandActuators`
  # remains the single owner of the §7 schedule that publication implies.
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

    @ Per-wheel tachometer readings (§8.5 momentum management). Guarded for the
    @ same reason `estimateIn` is: they are written on the producer's thread and
    @ read by `run`, and half a guarded pair is no mutual exclusion at all.
    @
    @ The wheel **speeds** arrive here, not the momentum: rotor inertia is a
    @ catalog fact this component carries as `WheelInertiaKgm2`, so the momentum
    @ the desaturation acts on is computed in exactly one place, from the same
    @ array geometry the allocation uses.
    guarded input port wheelSpeedIn: [GncMaxUnits] WheelSpeedMeasPort

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

    @ Ground override of the autonomous desaturation decision (§8.5). AUTO is the
    @ flown state: the controller engages the rods on its own momentum predicate
    @ inside POINT. FORCE and INHIBIT exist because a ground operator needs to be
    @ able to dump momentum ahead of a manoeuvre and to keep the rods quiet during
    @ a magnetically sensitive observation, and neither is a decision the vehicle
    @ can make for itself.
    enum DesatOverride : U8 {
      AUTO = 0 @< the momentum predicate decides
      FORCE = 1 @< desaturate whenever the mode and the field allow it
      INHIBIT = 2 @< never desaturate, whatever the momentum
    }

    @ Why the magnetorquer stuck-on monitor could not name a rod.
    enum StuckAttribution : U8 {
      DECISIVE = 0 @< exactly one rod carried a command; the EVR names it
      AMBIGUOUS = 1 @< several rods (or none) were commanded; only the mask is known
    }

    @ Why the wheel-momentum accounting produced no state this cycle (the FPP
    @ mirror of gnc::MomentumRefusal, so the EVR can carry the reason).
    enum MomentumRefusal : U8 {
      UNCONFIGURED = 0 @< the momentum configuration failed validation
      WHEEL_INVALID = 1 @< a wheel reported no usable (or no fresh) speed
      BAD_INPUT = 2 @< a wheel reported a non-finite speed
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

    @ Override the autonomous desaturation decision (§8.5). Accepted in every
    @ mode and at any time — it changes what the *next* cycle does, and a cycle
    @ that cannot desaturate anyway (wrong mode, no field, no momentum) simply
    @ does not, so FORCE is a permission and never a command to drive a rod
    @ blind. INHIBIT takes effect immediately, including mid-desaturation.
    guarded command CTRL_DESAT(
                                $action: DesatOverride @< AUTO / FORCE / INHIBIT
                              ) \
      opcode 3

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

    @ Slew-rate limit [rad/s] on the commanded rate (Kp/Kd)*dtheta, saturated by
    @ norm (Wie & Lu's rate-limited eigenaxis form). Outside it POINT is a
    @ constant-rate slew with pure rate damping, so a large error or a tumble
    @ handed to POINT is not a bang-bang manoeuvre with the wheels pinned.
    param PidMaxSlewRateRadps: F64

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

    # --- Wheel-drive friction feedforward (§8.5; REQ-ACTL-010) -------------
    #
    # A torque-mode drive delivers the commanded motor torque, but the body sees
    # the *net* rotor torque, so the allocation is wrong by the bearing friction
    # on every spinning wheel. On a loaded array that error is secular and, on
    # this vehicle, larger than the pointing integrator's whole authority. These
    # five parameters are the model that is fed forward to cancel it; the law is
    # `lib/gnc/rw_friction`, and it is applied per wheel after the allocation and
    # before the drive command.

    @ Enable the drive friction feedforward: 0 = off, 1 = on. Switchable for the
    @ same reason the disturbance-feedforward tiers are — flying the *same*
    @ vehicle with and without it is the only way to measure what it buys — and
    @ **the coefficients below are read and validated either way**, so a
    @ configuration missing its friction model is refused rather than quietly
    @ flown with the feedforward disabled.
    param WheelFrictionEnable: U8

    @ Coulomb (dry) friction torque magnitude [N*m] of one wheel, from its
    @ hardware-catalog entry (`dry_friction_nm`). One value, because the reference
    @ vehicle flies four identical wheels — the same reason WheelInertiaKgm2 and
    @ WheelMaxTorqueNm are scalars, and a mixed array needs all three per-unit.
    @ The config compiler checks this against the installed units' catalog values,
    @ so the number here cannot drift from the hardware it describes.
    param WheelDryFrictionNm: F64

    @ Viscous friction coefficient [N*m/(rad/s)] of one wheel, from its catalog
    @ entry (`viscous_friction_nm_s`). Continuous through zero speed, so it needs
    @ no deadband; compensated because leaving out a modelled, exactly-known term
    @ would be arbitrary, not because it is large (~5e-6 N*m at 1 rad/s here).
    param WheelViscousFrictionNmS: F64

    @ Blend half-width [rad/s] for the Coulomb term's sign. Friction is
    @ -sgn(omega)*tau_c, and a feedforward built on a bare sign() chatters at low
    @ wheel speed — injecting a square wave of amplitude 2*tau_c at exactly the
    @ operating point where the vehicle is otherwise quietest. The compensation
    @ therefore ramps linearly from zero at omega = 0 to full magnitude at
    @ |omega| = this. **The cost is stated rather than hidden**: inside the band
    @ the friction is deliberately under-compensated. It is bounded above by the
    @ vehicle's own wheel-speed range — a band covering a large fraction of the
    @ speeds the array actually runs at under-compensates across the flight
    @ envelope rather than only at a zero crossing — and below by the
    @ tachometer's resolution and noise, since under those the *sign* of omega is
    @ not a measurement. Must be positive; zero is the discontinuous sign() this
    @ exists to avoid and is refused.
    param WheelFrictionDeadbandRadps: F64

    @ Per-wheel trim on the modelled friction, dimensionless, in vehicle build
    @ order. **Policy: at or below 1.** Feedforward against a modelled disturbance
    @ helps monotonically only while the model does not exceed the truth —
    @ compensating a fraction k <= 1 leaves (1-k) of the friction, same sign,
    @ never worse — whereas over-compensating reverses the residual's sign and
    @ past k = 2 makes it larger than doing nothing. A flight campaign that
    @ measures the real rundown trims *down* toward it. The value is **not**
    @ clamped in flight: silently rewriting a commanded trim would hide the one
    @ case an operator needs to see. Zero disables compensation on that wheel.
    param WheelFrictionScale: F64PerUnit

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

    # --- Momentum management and desaturation (§8.5) -----------------------

    @ Rotor inertia of one reaction wheel [kg*m^2], from its catalog entry. Turns
    @ the tachometer reading into stored momentum; one value, because the
    @ reference vehicle flies four identical wheels (a mixed array needs a
    @ per-unit array here, as `WheelMaxTorqueNm` does).
    param WheelInertiaKgm2: F64

    @ Body-frame inertia tensor **diagonal** [kg*m^2]. Needed for two things the
    @ pointing law alone did not need: the total system momentum
    @ H = J*omega + h_wheels the disturbance observer differences, and the
    @ gravity-gradient feedforward. The off-diagonal terms are taken as zero —
    @ the same principal-axis assumption the §8.5 linear analysis makes, and which
    @ `analysis.control.vehicle.load_vehicle` refuses a config for if it is false,
    @ so the two cannot quietly disagree about which vehicle they describe.
    param InertiaBodyKgm2: Vec3F64

    @ Momentum bias to hold [N*m*s], body frame. Zero for a zero-momentum vehicle.
    param MomentumTargetBody: Vec3F64

    @ ||h - h_target|| [N*m*s] above which desaturation is required.
    param MomentumDesatEnterNms: F64

    @ ||h - h_target|| [N*m*s] below which, held for MomentumDesatConfirmCycles,
    @ the desaturation ends. Must be below MomentumDesatEnterNms: the deadband is
    @ what stops the rods chattering on and off at the threshold.
    param MomentumDesatExitNms: F64

    @ Consecutive cycles under MomentumDesatExitNms that end a desaturation.
    param MomentumDesatConfirmCycles: U32

    @ Stored-momentum ceiling [N*m*s] the §9 envelope monitor watches. **Not** the
    @ wheels' capacity: it is the bound the vehicle's control margins are analysed
    @ at (§8.5 SISO validity boundary — above it the gyroscopic coupling makes the
    @ per-axis margin analysis describe a different vehicle), which is a far
    @ smaller number and the one worth alarming on. Must be at or above
    @ MomentumDesatEnterNms.
    param MomentumEnvelopeNms: F64

    @ One wheel's momentum capacity [N*m*s] (the catalog's max_momentum_nms; the
    @ config compiler cross-checks it). The §9 per-wheel monitor watches the
    @ largest single-wheel momentum against it — a number the body-momentum
    @ envelope cannot see, because a four-wheel array can carry momentum in its
    @ null space (wheels spinning against each other) that sums to nothing in
    @ body axes while one wheel walks to its stop.
    param WheelCapacityNms: F64

    @ Cross-product desaturation gain k_d [1/s] in m = k_d (dh x B)/||B||^2. The
    @ perpendicular momentum error decays with time constant 1/k_d while the rods
    @ are unsaturated — the duty division inside the law is what keeps the duty
    @ factor out of the decay rate.
    param DesatGainPerSec: F64

    # --- Disturbance feedforward (§8.5 tiers 1-2) --------------------------

    @ Enable the tier-1 model-based feedforward (gravity gradient + residual
    @ dipole): 0 = off, 1 = on. Separately switchable from the observer because
    @ they fail differently — a wrong inertia or a wrong residual dipole makes
    @ tier 1 harmful while tier 2 is still sound, and vice versa.
    param FeedforwardModelEnable: U8

    @ Enable the tier-2 momentum-based observer's contribution to the feedforward:
    @ 0 = off, 1 = on. The observer itself **runs regardless**, because it is also
    @ the §9 momentum-anomaly monitor and a monitor that can be switched off by a
    @ control-tuning parameter is not a monitor.
    param FeedforwardObserverEnable: U8

    @ Low-pass time constant [s] of the residual-torque observer. Far above the
    @ control period, so the momentum difference quotient's noise averages down,
    @ and far below the orbital period, so a real secular torque is still tracked.
    param ObserverTauSec: F64

    @ The vehicle's body-fixed residual magnetic moment [A*m^2] for the tier-1
    @ `m_res x B` feedforward — the magnetic-cleanliness allocation from §19.1,
    @ not a fitted state. Fitting it from the flight data is §8.5 tier 3.
    param ResidualDipoleAm2: Vec3F64

    @ Unmodelled secular torque [N*m] above which the §9 momentum anomaly is
    @ declared. A **budget** value with declared margin over the modelled
    @ disturbance environment, never a number fitted to a measurement.
    param DisturbanceBudgetNm: F64

    @ Consecutive observer updates over DisturbanceBudgetNm that latch the
    @ anomaly, and consecutive updates under DisturbanceClearNm that clear it.
    param DisturbanceAnomalyCycles: U32

    @ ||tau|| [N*m] below which, held for DisturbanceAnomalyCycles consecutive
    @ updates, the §9 momentum anomaly clears. Must be at or below
    @ DisturbanceBudgetNm: the deadband is what stops an estimate parked at the
    @ budget — which is what a real fault at the margin looks like through the
    @ observer's low-pass — from cycling the anomaly once per confirmation count.
    param DisturbanceClearNm: F64

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

    @ Per-wheel commanded torque [N*m], in vehicle build order. This is what the
    @ drives are told — the allocation's demand **plus** the friction feedforward.
    telemetry WheelTorque: F64PerUnit

    @ Friction feedforward actually applied per wheel [N*m], in vehicle build
    @ order, after the torque-box clamp. NaN on a wheel with no usable tachometer
    @ (no speed, no sign, no feedforward — that wheel keeps the uncompensated
    @ behaviour), and NaN on every wheel when the feedforward is disabled or the
    @ law refused. Subtract it from WheelTorque to recover the allocation's
    @ demand, which is what makes the two channels together an ablation an
    @ operator can read without a ground model.
    telemetry WheelFrictionNm: F64PerUnit

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

    @ Stored wheel-array momentum in body axes [N*m*s]. No value when a wheel
    @ reported no usable speed — the sum needs every term.
    telemetry StoredMomentum: Vec3F64

    @ ||h|| [N*m*s], the quantity MomentumEnvelopeNms gates.
    telemetry StoredMomentumNms: F64

    @ Largest single-wheel momentum max_i |I_w w_i| [N*m*s], the quantity the
    @ WheelCapacityNms monitor gates. Not bounded by StoredMomentumNms.
    telemetry MaxWheelMomentumNms: F64

    @ Norm of the wheel-momentum vector's null-space component [N*m*s]: momentum
    @ the wheels hold against each other that produces no body momentum. Zero on
    @ a three-wheel array; growth here is a wheel drifting toward its stop where
    @ no body-momentum threshold will see it.
    telemetry NullSpaceMomentumNms: F64

    @ The wheel-momentum accounting produced a usable state this cycle. False
    @ means desaturation and the §9 envelope and momentum-anomaly monitors are
    @ all running blind — the condition MomentumUnavailable events on.
    telemetry MomentumValid: bool

    @ A desaturation is in progress: the rods are being driven to unload the
    @ wheels while the wheels hold the attitude.
    telemetry DesatActive: bool

    @ Ground override state of the desaturation decision.
    telemetry DesatOverrideTlm: DesatOverride

    @ Observed unmodelled secular external torque [N*m], body axes — the §8.5
    @ tier-2 estimate, which is also what the §9 anomaly monitor gates. No value
    @ until the observer has accepted its first update.
    telemetry ResidualTorque: Vec3F64

    @ Feedforward torque [N*m] added to this cycle's demand: minus the modelled
    @ and observed disturbance, as enabled. Computed in every mode (the observer
    @ behind it is also the §9 monitor); consumed by POINT.
    telemetry FeedforwardTorque: Vec3F64

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

    @ Desaturation engaged: the rods are now unloading the wheels while POINT
    @ holds the attitude. Both edges are events because a desaturation is a
    @ magnetic activity an operator correlating a payload anomaly needs to see the
    @ start and the end of.
    event DesatEngaged(momentumNms: F64, thresholdNms: F64, forced: bool) \
      severity activity high \
      format "Desaturation engaged at {} N*m*s (threshold {}, forced {})"

    @ Desaturation disengaged: the momentum error has been under the exit
    @ threshold for the confirmation count, or the mode/override withdrew the
    @ permission.
    event DesatDisengaged(momentumNms: F64) \
      severity activity high \
      format "Desaturation disengaged at {} N*m*s"

    @ Stored momentum is above MomentumEnvelopeNms (§9). Not a wheel-capacity
    @ alarm — it means the vehicle has left the momentum range its pointing
    @ margins were analysed over (§8.5 SISO validity boundary), so the loop is no
    @ longer certified even though every wheel is comfortable.
    event MomentumEnvelopeExceeded(momentumNms: F64, envelopeNms: F64) \
      severity warning high \
      format "Stored momentum {} N*m*s is outside the {} N*m*s envelope"

    @ Stored momentum is back inside the envelope. The recovery edge, on the same
    @ comparison that raised it.
    event MomentumEnvelopeRecovered(momentumNms: F64) \
      severity activity high \
      format "Stored momentum back inside the envelope at {} N*m*s"

    @ A wheel is above 90 % of WheelCapacityNms (§9). Edge-gated on the crossing.
    @ This is the per-wheel alarm the envelope is not: it fires whether the
    @ momentum is body momentum or null-space momentum the body cannot see.
    event WheelNearCapacity(maxWheelNms: F64, capacityNms: F64, nullSpaceNms: F64) \
      severity warning high \
      format "A wheel holds {} N*m*s of its {} N*m*s capacity ({} N*m*s of it in the null space)"

    @ Every wheel is back under 90 % of its capacity. The recovery edge.
    event WheelCapacityRecovered(maxWheelNms: F64) \
      severity activity high \
      format "Largest wheel momentum back to {} N*m*s"

    @ The observed unmodelled secular torque has been outside the modelled
    @ disturbance budget for DisturbanceAnomalyCycles consecutive updates (§9).
    @ The signature of a torque source the vehicle does not model — a stuck
    @ thruster, an unlatched deployment, a residual dipole far past its
    @ allocation. Reported, not acted on: the response is the Phase-7 state
    @ machine's, and the honest action here is to name it.
    event MomentumAnomaly(torqueNm: F64, budgetNm: F64) \
      severity warning high \
      format "Momentum anomaly: {} N*m of unmodelled secular torque against a {} N*m budget"

    @ The observed torque fell below DisturbanceClearNm for the same number of
    @ consecutive updates that latched the anomaly.
    event MomentumAnomalyCleared(torqueNm: F64) \
      severity activity high \
      format "Momentum anomaly cleared at {} N*m"

    @ The wheel-momentum accounting refused this cycle — one dead or stale
    @ tachometer is enough, since the stored momentum is a sum that needs every
    @ term. While it persists, desaturation and the §9 envelope and
    @ momentum-anomaly monitors are all running blind, which is why the refusal
    @ is an event and not just a NaN on a strip chart: a monitor that loses its
    @ input has to say so, at the bounded AlertCycles cadence. `wheel` names the
    @ refusing wheel when the reason is per-wheel, and is -1 otherwise.
    event MomentumUnavailable(reason: MomentumRefusal, wheel: I32, cycles: U32) \
      severity warning high \
      format "Momentum accounting unavailable: {} (wheel {}, {} cycles)"

    @ CTRL_DESAT changed the ground override.
    event DesatOverrideChanged($action: DesatOverride) \
      severity activity high \
      format "Desaturation override set to {}"

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
