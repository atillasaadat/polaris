module flight {

  @ SITL lockstep bridge: the flight end of the §2.2/§2.4 plant<->FSW transport.
  @
  @ Sits at the payload boundary of a dedicated SITL comm stack (its own
  @ Drv.TcpClient + Svc.FrameAccumulator + Svc.FprimeDeframer / Svc.FprimeFramer
  @ instances, disjoint from the GDS ground link). Deframed SITL payloads
  @ (lib/sitl/wire.hpp) arrive on dataIn; the component validates and answers on
  @ dataOut, which the framer wraps and the same TcpClient returns to the sim.
  @
  @ Each STEP_REQ synchronously drives the FSW's 10 Hz cycle before the reply is
  @ built (design doc §2.4 steps 3-4): dataIn decodes the barrier request, pushes
  @ the sim epoch to the time provider (timeSetOut), fires the SITL rate group
  @ (sitlCycleOut -> Svc.PassiveRateGroup, run-to-completion on this task), then
  @ assembles the STEP_REPLY from the actuator commands the rate group latched
  @ back on wheelCmdIn/mtqCmdIn. Passive component because the whole
  @ uplink->cycle->reply path runs synchronously on the TcpClient receive task;
  @ no queue is needed. The synchronous reply cannot deadlock against the sim's
  @ blocking send only while both messages stay far below the TCP socket buffers
  @ (STEP_REQ <= ~3.8 KB, STEP_REPLY <= 336 B) -- revisit if the reply ever grows
  @ toward that scale (command values changed but not the reply size).
  @
  @ Decode/reply logic lives in lib/sitl/handler.hpp (SitlHandler) so the byte
  @ protocol is unit-tested without a topology.
  passive component SitlBridge {

    # ----------------------------------------------------------------------
    # SITL payload ports (own comm stack, not the GDS ComCcsds stack)
    # ----------------------------------------------------------------------

    @ Deframed SITL payload in, from the SITL FprimeDeframer.dataOut
    sync input port dataIn: Svc.ComDataWithContext

    @ Returns ownership of the received payload buffer to the deframer
    output port dataReturnOut: Svc.ComDataWithContext

    @ Reply payload out, to the SITL FprimeFramer.dataIn
    output port dataOut: Svc.ComDataWithContext

    @ Receives back ownership of the reply buffer after framing (a fixed
    @ internal buffer, so nothing is deallocated)
    sync input port dataReturnIn: Svc.ComDataWithContext

    # ----------------------------------------------------------------------
    # SITL rate-group coupling (design doc §2.4 steps 3-4)
    # ----------------------------------------------------------------------

    @ Drives the SITL PassiveRateGroup synchronously, once per STEP, between
    @ decoding the barrier request and building the reply. The Os.RawTime this
    @ port carries is a wall-clock read the rate group uses only for its
    @ CycleTime/MaxCycleTime diagnostics -- those two channels are exempt from
    @ sim-time determinism (never reproducible run-to-run); nothing on the
    @ command/reply path consumes them.
    output port sitlCycleOut: Svc.Cycle

    @ Publishes the macro-step sim epoch to the SITL time provider, before the
    @ rate group fires, so FSW time keys off sim time
    output port timeSetOut: SitlTimeSet

    @ Latest reaction-wheel torque commands from the rate group, latched for the
    @ next STEP_REPLY
    sync input port wheelCmdIn: WheelTorqueCmd

    @ Latest magnetorquer dipole commands from the rate group, latched for the
    @ next STEP_REPLY
    sync input port mtqCmdIn: MtqDipoleCmd

    # ----------------------------------------------------------------------
    # Sensor measurement outputs (the SITL end of the GncPorts seam, §8.0)
    # ----------------------------------------------------------------------
    #
    # Each STEP_REQ's per-unit sensor records are republished on these arrays
    # before the rate group is cycled, so the GNC components see this step's
    # measurements in the very cycle the barrier drives. Unit i of the wire
    # message goes to port i (positional identity, §19.4 vehicle build order);
    # ports past the HELLO-declared count are never called. On hardware these
    # come from Drv sensor drivers instead and this component is not built —
    # which is exactly why the seam is a shared port module.

    @ Per-IMU delta-angle/delta-velocity increments for this macro step.
    output port imuOut: [GncMaxUnits] ImuMeasPort

    @ Per-sun-sensor processed unit vectors.
    output port sunSensorOut: [GncMaxUnits] SunSensorMeasPort

    @ Per-magnetometer field measurements.
    output port magnetometerOut: [GncMaxUnits] MagnetometerMeasPort

    @ Per-receiver GNSS PVT fixes.
    output port gnssOut: [GncMaxUnits] GnssMeasPort

    @ Per-tracker attitude solutions (no coarse-mode consumer; fine mode is the
    @ MEKF push).
    output port starTrackerOut: [GncMaxUnits] StarTrackerMeasPort

    # ----------------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------------

    @ Macro steps exchanged with the sim since connect
    telemetry MacroStep: U64

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ First valid SITL message decoded since start: the sim link is live
    event SitlConnected \
      severity activity high \
      format "SITL link connected: first valid message received"

    @ HELLO received; the per-type unit counts that size every STEP exchange
    event HelloReceived(
        nImu: U32,
        nStarTracker: U32,
        nSunSensor: U32,
        nMagnetometer: U32,
        nGnss: U32,
        nWheel: U32,
        nMtq: U32
      ) \
      severity activity high \
      format "SITL HELLO: imu={} st={} ss={} mag={} gnss={} wheel={} mtq={}"

    @ Periodic progress marker at STEP milestones
    event StepMilestone(macroStep: U64) \
      severity activity low \
      format "SITL macro step {}"

    @ A malformed or out-of-order SITL message was rejected (no reply sent)
    event MalformedMessage(msgType: U16, bytes: U32) \
      severity warning high \
      format "SITL message rejected: type={} bytes={}"

    @ SHUTDOWN received; the bridge goes quiet (stops answering)
    event SitlShutdown \
      severity activity high \
      format "SITL SHUTDOWN received: bridge quiescent"

    # ----------------------------------------------------------------------
    # Standard AC ports
    # ----------------------------------------------------------------------

    @ Port for requesting the current time
    time get port timeCaller

    @ Port for sending textual representation of events
    text event port logTextOut

    @ Port for sending events to downlink
    event port logOut

    @ Port for emitting telemetry
    telemetry port tlmOut

  }

}
