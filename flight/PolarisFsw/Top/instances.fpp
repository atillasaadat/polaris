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

  # Onboard time/EOP/ephemeris table provider (design doc §11.3, §22). Loads the
  # leap/EOP/Chebyshev tables at setup and serves them to the Phase-4 GNC stack;
  # a flight component (ships to hardware), not SITL infrastructure.
  instance onboardTables: flight.OnboardTables base id 0x10020000

  # Coarse attitude estimator (design doc §8.1, §10). A flight component: it runs
  # on the GNC rate group, consumes the GncPorts measurement seam (fed by
  # SitlBridge under SITL, by Drv sensor drivers on the vehicle) and the
  # OnboardTables query ports, and publishes the §8.0 estimate.
  instance attitudeEstimator: flight.AttitudeEstimator base id 0x10030000

  instance rateGroupDriver: Svc.RateGroupDriver base id 0x10011000

  instance systemResources: Svc.SystemResources base id 0x10012000

  instance timer: Svc.LinuxTimer base id 0x10013000

  instance comDriver: Drv.TcpClient base id 0x10014000

  # ----------------------------------------------------------------------
  # SITL lockstep transport (design doc §2.2, §2.4)
  # ----------------------------------------------------------------------
  #
  # The plant<->FSW SITL comm stack (sitlBridge, sitlRateGroup, scriptedCmdSource
  # and their dedicated TcpClient/ComStub/FrameAccumulator/deframer/framer/
  # bufferManager) now lives in the PolarisSitl subtopology
  # (flight/PolarisFsw/PolarisSitl/), imported by topology.fpp. Base IDs are
  # preserved there (0x10015000 + offsets) so the dictionary is unchanged.
  #
  # SitlTime (above) intentionally stays here: it is the deployment-wide time
  # source served to every component via `time connections`, not SITL-only
  # infrastructure — only its SITL activation is gated at runtime.

}
