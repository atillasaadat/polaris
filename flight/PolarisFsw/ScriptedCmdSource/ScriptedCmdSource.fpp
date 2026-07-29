module flight {

  @ Phase-4 placeholder actuator command source on the SITL rate group.
  @
  @ Until the Phase-4 GNC estimator/controller components exist, this stands in
  @ as the rate group's actuator commander (design doc §2.4 step 4). Driven once
  @ per macro step by the SITL PassiveRateGroup, it reads sim time from its time
  @ port (served by SitlTime) and emits a deterministic, pure-function-of-sim-time
  @ profile (lib/sitl/scripted_profile.hpp) to SitlBridge, which latches it into
  @ the STEP_REPLY. Disabled by default (commands zero), enabled via setEnabled()
  @ so the existing zero-command SITL gate is unaffected; the nonzero profile is a
  @ closed-loop exercise, not a control law. Replaced wholesale by real GNC.
  passive component ScriptedCmdSource {

    @ Rate-group member entry: compute and emit this step's commands
    sync input port run: Svc.Sched

    @ Reaction-wheel torque commands out, to SitlBridge
    output port wheelCmdOut: WheelTorqueCmd

    @ Magnetorquer dipole commands out, to SitlBridge
    output port mtqCmdOut: MtqDipoleCmd

    # ----------------------------------------------------------------------
    # Telemetry & events
    # ----------------------------------------------------------------------

    @ Rate-group cycles executed since start
    telemetry CyclesRun: U32

    @ Whether the scripted profile is emitting nonzero commands
    telemetry Enabled: bool

    @ Emitted once at configuration: whether the scripted profile is enabled
    event Configured(enabled: bool) \
      severity activity high \
      format "ScriptedCmdSource configured: enabled={}"

    # ----------------------------------------------------------------------
    # Standard AC ports
    # ----------------------------------------------------------------------

    @ Port for requesting the current time (served by SitlTime = sim time in SITL)
    time get port timeCaller

    @ Port for sending textual representation of events
    text event port logTextOut

    @ Port for sending events to downlink
    event port logOut

    @ Port for emitting telemetry
    telemetry port tlmOut

  }

}
