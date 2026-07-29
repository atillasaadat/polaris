module flight {

  # ----------------------------------------------------------------------
  # Base ID Convention
  # ----------------------------------------------------------------------
  #
  # All Base IDs follow the 8-digit hex format: 0xDSSCCxxx
  #
  # Where:
  #   D   = Deployment digit (1 for this deployment)
  #   SS  = Subtopology digits (00 for main topology, 01-05 for subtopologies)
  #   CC  = Component digits (00, 01, 02, etc.)
  #   xxx = Reserved for internal component items (events, commands, telemetry)
  #

  # ----------------------------------------------------------------------
  # Defaults
  # ----------------------------------------------------------------------

  module Default {
    constant QUEUE_SIZE = 10
    constant STACK_SIZE = 64 * 1024
  }

  # ----------------------------------------------------------------------
  # Active component instances
  # ----------------------------------------------------------------------

  instance rateGroup1: Svc.ActiveRateGroup base id 0x10001000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 43

  instance rateGroup2: Svc.ActiveRateGroup base id 0x10002000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 42

  instance rateGroup3: Svc.ActiveRateGroup base id 0x10003000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 41

  instance cmdSeq: Svc.CmdSequencer base id 0x10004000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 40

  # ----------------------------------------------------------------------
  # Queued component instances
  # ----------------------------------------------------------------------


  # ----------------------------------------------------------------------
  # Passive component instances
  # ----------------------------------------------------------------------

  # SITL sim-time provider — the deployment's single time source (design doc
  # §2.4, §3.2). Replaces Svc.ChronoTime: serves sim time in a SITL run, and the
  # same workstation wall clock as ChronoTime otherwise.
  instance sitlTime: flight.SitlTime base id 0x10010000

  instance rateGroupDriver: Svc.RateGroupDriver base id 0x10011000

  instance systemResources: Svc.SystemResources base id 0x10012000

  instance timer: Svc.LinuxTimer base id 0x10013000

  instance comDriver: Drv.TcpClient base id 0x10014000

  # ----------------------------------------------------------------------
  # SITL lockstep transport (design doc §2.2, §2.4)
  # ----------------------------------------------------------------------
  #
  # A dedicated comm stack for the plant<->FSW TCP link, disjoint from the GDS
  # ComCcsds stack above: its own TcpClient + ComStub + FrameAccumulator +
  # FprimeDeframer/FprimeFramer, feeding the SitlBridge payload boundary. All
  # inert unless --sitl-port is given (the TcpClient is never started).

  instance sitlBridge: flight.SitlBridge base id 0x10015000

  # SITL rate group (passive → runs to completion on the SITL receive task, so
  # the 10 Hz FSW cycle is deterministic and barrier-driven, not wall-clock
  # driven). Cycled only by sitlBridge.sitlCycleOut, never by rateGroupDriver.
  instance sitlRateGroup: Svc.PassiveRateGroup base id 0x1001C000

  # Phase-4 placeholder actuator commander; the SITL rate group's only member
  # until real GNC components replace it.
  instance scriptedCmdSource: flight.ScriptedCmdSource base id 0x1001D000

  instance comDriverSitl: Drv.TcpClient base id 0x10016000

  instance comStubSitl: Svc.ComStub base id 0x10017000

  instance frameAccumulatorSitl: Svc.FrameAccumulator base id 0x10018000

  instance deframerSitl: Svc.FprimeDeframer base id 0x10019000

  instance framerSitl: Svc.FprimeFramer base id 0x1001A000

  instance commsBufferManagerSitl: Svc.BufferManager base id 0x1001B000

}
