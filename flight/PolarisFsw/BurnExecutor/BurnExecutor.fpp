module flight {

  # ----------------------------------------------------------------------
  # Finite-burn executor (design doc §17, §8.3; REQ-MAN-001)
  # ----------------------------------------------------------------------
  #
  # The seam the Phase-8 targeting will command, built ahead of it (Push 70):
  # a burn is a throttle held on the configured thrusters for a duration, and
  # while it is held the executor tells the orbit filter what acceleration is
  # acting — commanded thrust over its own mass estimate, rotated to ECI with
  # the current attitude — so the filter propagates through the burn instead
  # of rejecting every fix under it (§8.3, the Ceresoli et al. 2025 result:
  # a coast through a burn without the thrust known is a kilometre-class
  # error the filter cannot justify).
  #
  # **Every burn is finite (§17 decision).** There is no impulsive path here:
  # the thruster is commanded on, held, and commanded off; the sim's thruster
  # model applies its own rise/fall, thrust error and misalignment; what the
  # filter is told is the *commanded* acceleration with a knowledge fraction as
  # its sigma. Steering (velocity-tracking, Delta-V mode) and targeting are
  # Phase 8: today the thrust direction is the mounted axis and pointing the
  # vehicle is the §8.5 controller's job.
  #
  # **Rate-group member 3, after the controller.** Its accel output is latched
  # by the orbit estimator (member 0) for the *next* cycle — which is also the
  # step over which the throttle command, riding the same STEP_REPLY, first acts
  # on the plant. The two are therefore aligned by construction, not by luck.
  #
  # **Passive, guarded**: `run` on the rate group's thread, `attitudeIn` on the
  # estimator's, commands on the dispatcher's — every handler that touches the
  # burn state or the latched attitude is `guarded`; the `-b` bench hook runs
  # the command body from inside `run` (the ports share the mutex).
  passive component BurnExecutor {

    @ Executor cycle at the GNC rate (10 Hz, §2.4).
    guarded input port run: Svc.Sched

    @ The attitude estimate (Body <- ECI), latched; the burn's acceleration is
    @ rotated to ECI with it. Invalid or stale refuses/aborts a burn.
    guarded input port attitudeIn: AttitudeEstimatePort

    @ Per-thruster throttles, held for the next macro step (SitlBridge under
    @ SITL; a valve driver on hardware). Zero when idle.
    output port thrusterCmdOut: ThrusterThrottleCmd

    @ The commanded non-gravitational acceleration, published every cycle:
    @ valid=true while burning, valid=false while idle so the filter sees "no
    @ thrust" explicitly rather than a stale vector.
    output port accelOut: NonGravAccelPort

    enum BurnState : U8 {
      IDLE = 0
      BURNING = 1
      ABORTED = 2 @< the last burn ended by abort; cleared by the next start
    }

    enum BurnRefusal : U8 {
      UNCONFIGURED = 0
      DURATION = 1 @< non-positive or above MaxBurnDurationS
      THROTTLE = 2 @< outside (0, 1]
      ATTITUDE = 3 @< no valid attitude, or older than MaxAttitudeAgeS
      ALREADY_BURNING = 4
    }

    @ Fire the configured thrusters at `throttle` for `durationS`.
    guarded command BURN_START(
                                durationS: F64 @< burn duration [s]
                                throttleFrac: F64 @< throttle fraction, 0 exclusive to 1 inclusive
                              ) \
      opcode 0

    @ Stop a burn now; the thrusters are commanded off this cycle.
    guarded command BURN_ABORT \
      opcode 1

    # --- Parameters (§19.3 — no defaults; a missing value refuses) ----------

    @ Installed thrusters, 1..GncMaxUnits (the config compiler checks the count).
    param ThrusterCount: U32

    @ Nominal thrust direction of each thruster in the body frame, flat
    @ [x0,y0,z0, x1,...] in vehicle build order (checked against each unit's
    @ `thrust_axis`). The direction the force acts on the vehicle.
    param ThrusterAxesBody: Vec3F64PerUnit

    @ Rated thrust of each thruster [N] at throttle 1 (catalog `thrust_n`).
    param ThrusterThrustN: F64PerUnit

    @ Specific impulse of each thruster [s] (catalog `isp_s`), for the onboard
    @ mass-depletion estimate.
    param ThrusterIspS: F64PerUnit

    @ Vehicle wet mass at configuration [kg]; the executor depletes its own
    @ estimate m -= sum(F_i/(Isp_i g0)) dt while burning and telemeters it.
    param VehicleMassKg: F64

    @ 1-sigma knowledge of the commanded thrust as a fraction of it; reported
    @ to the orbit filter as the acceleration's sigma.
    param ThrustKnowledgeFrac: F64

    @ Longest burn a single command may ask for [s].
    param MaxBurnDurationS: F64

    @ Oldest attitude estimate a burn may be started or continued on [s].
    param MaxAttitudeAgeS: F64

    # --- Telemetry ----------------------------------------------------------

    telemetry BurnStateTlm: BurnState
    telemetry BurnRemainingS: F64
    telemetry BurnDeltaVMps: F64
    telemetry MassEstimateKg: F64
    telemetry ThrottleCmd: F64

    # --- Events -------------------------------------------------------------

    event BurnStarted(durationS: F64, throttleFrac: F64) \
      severity activity high \
      format "Burn started: {} s at throttle {}"

    event BurnCompleted(deltaVMps: F64) \
      severity activity high \
      format "Burn completed: {} m/s accumulated"

    event BurnAborted(reason: BurnRefusal) \
      severity warning high \
      format "Burn aborted: {}"

    event BurnRefused(reason: BurnRefusal) \
      severity warning low \
      format "Burn refused: {}"

    event ConfigInvalid(detail: string size 80) \
      severity warning high \
      format "Burn executor configuration invalid: {}"

    # --- Standard AC ports --------------------------------------------------

    time get port timeCaller
    param get port prmGetOut
    param set port prmSetOut
    command reg port cmdRegOut
    command recv port cmdIn
    command resp port cmdResponseOut
    event port logOut
    text event port logTextOut
    telemetry port tlmOut
  }
}
