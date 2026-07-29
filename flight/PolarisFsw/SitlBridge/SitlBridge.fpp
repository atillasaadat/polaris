module flight {

  @ SITL lockstep bridge: the flight end of the §2.2/§2.4 plant<->FSW transport.
  @
  @ Sits at the payload boundary of a dedicated SITL comm stack (its own
  @ Drv.TcpClient + Svc.FrameAccumulator + Svc.FprimeDeframer / Svc.FprimeFramer
  @ instances, disjoint from the GDS ground link). Deframed SITL payloads
  @ (lib/sitl/wire.hpp) arrive on dataIn; the component validates and answers on
  @ dataOut, which the framer wraps and the same TcpClient returns to the sim.
  @
  @ THIS PUSH the bridge answers autonomously with zero actuator commands: it is
  @ not yet wired to the control rate group (that is the next push). Passive
  @ component because the whole uplink->reply path runs synchronously on the
  @ TcpClient receive task; no queue is needed. The synchronous reply cannot
  @ deadlock against the sim's blocking send only while both messages stay far
  @ below the TCP socket buffers (STEP_REQ <= ~3.8 KB, STEP_REPLY <= 336 B
  @ today) -- revisit if the reply ever grows toward that scale.
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
