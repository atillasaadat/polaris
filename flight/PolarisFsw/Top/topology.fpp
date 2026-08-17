module flight {

  # ----------------------------------------------------------------------
  # Symbolic constants for port numbers
  # ----------------------------------------------------------------------

  enum Ports_RateGroups {
    rateGroup1
    rateGroup2
    rateGroup3
  }

  topology PolarisFsw {

  # ----------------------------------------------------------------------
  # Subtopology imports
  # ----------------------------------------------------------------------
    import CdhCore.Subtopology
    import ComCcsds.Subtopology
    import DataProducts.Subtopology
    import FileHandling.Subtopology
    # SITL lockstep transport (design doc §2.2). A hardware build drops this
    # import and the Sitl connections block below (recipe: PolarisFsw/README.md).
    import PolarisSitl.Subtopology

  # ----------------------------------------------------------------------
  # Instances used in the topology
  # ----------------------------------------------------------------------
    instance sitlTime
    instance onboardTables
    instance orbitEstimator
    instance attitudeEstimator
    instance attitudeController
    instance rateGroup1
    instance rateGroup2
    instance rateGroup3
    instance rateGroupDriver
    instance systemResources
    instance timer
    instance comDriver
    instance cmdSeq

    # SITL lockstep transport instances come from the imported PolarisSitl
    # subtopology; only sitlTime (above) lives in this topology.

  # ----------------------------------------------------------------------
  # Pattern graph specifiers
  # ----------------------------------------------------------------------

    command connections instance CdhCore.cmdDisp
    event connections instance CdhCore.events
    telemetry connections instance CdhCore.tlmSend
    text event connections instance CdhCore.textLogger
    health connections instance CdhCore.$health
    param connections instance FileHandling.prmDb
    time connections instance sitlTime

  # ----------------------------------------------------------------------
  # Telemetry packets (only used when TlmPacketizer is used)
  # ----------------------------------------------------------------------

    # include "PolarisFswPackets.fppi"

  # ----------------------------------------------------------------------
  # Direct graph specifiers
  # ----------------------------------------------------------------------

    connections ComCcsds_CdhCore {
      # Core events and telemetry to communication queue
      CdhCore.events.PktSend -> ComCcsds.comQueue.comPacketQueueIn[ComCcsds.Ports_ComPacketQueue.EVENTS]
      CdhCore.tlmSend.PktSend -> ComCcsds.comQueue.comPacketQueueIn[ComCcsds.Ports_ComPacketQueue.TELEMETRY]

      # Router to Command Dispatcher
      ComCcsds.fprimeRouter.commandOut -> CdhCore.cmdDisp.seqCmdBuff
      CdhCore.cmdDisp.seqCmdStatus -> ComCcsds.fprimeRouter.cmdResponseIn

    }

    connections ComCcsds_FileHandling {
      # File Downlink to Communication Queue
      FileHandling.fileDownlink.bufferSendOut -> ComCcsds.comQueue.bufferQueueIn[ComCcsds.Ports_ComBufferQueue.FILE]
      ComCcsds.comQueue.bufferReturnOut[ComCcsds.Ports_ComBufferQueue.FILE] -> FileHandling.fileDownlink.bufferReturn

      # Router to File Uplink
      ComCcsds.fprimeRouter.fileOut -> FileHandling.fileUplink.bufferSendIn
      FileHandling.fileUplink.bufferSendOut -> ComCcsds.fprimeRouter.fileBufferReturnIn
    }

    connections Communications {
      # ComDriver buffer allocations
      comDriver.allocate      -> ComCcsds.commsBufferManager.bufferGetCallee
      comDriver.deallocate    -> ComCcsds.commsBufferManager.bufferSendIn

      # ComDriver <-> ComStub (Uplink)
      comDriver.$recv                     -> ComCcsds.comStub.drvReceiveIn
      ComCcsds.comStub.drvReceiveReturnOut -> comDriver.recvReturnIn

      # ComStub <-> ComDriver (Downlink)
      ComCcsds.comStub.drvSendOut      -> comDriver.$send
      comDriver.ready         -> ComCcsds.comStub.drvConnected
    }

    connections FileHandling_DataProducts {
      # Data Products to File Downlink
      DataProducts.dpCat.fileOut -> FileHandling.fileDownlink.SendFile
      FileHandling.fileDownlink.FileComplete -> DataProducts.dpCat.fileDone
    }

    connections RateGroups {
      # timer to drive rate group
      timer.CycleOut -> rateGroupDriver.CycleIn

      # Rate group 1
      rateGroupDriver.CycleOut[Ports_RateGroups.rateGroup1] -> rateGroup1.CycleIn
      rateGroup1.RateGroupMemberOut[0] -> CdhCore.tlmSend.Run
      rateGroup1.RateGroupMemberOut[1] -> FileHandling.fileDownlink.Run
      rateGroup1.RateGroupMemberOut[2] -> systemResources.run
      rateGroup1.RateGroupMemberOut[3] -> ComCcsds.comQueue.run
      rateGroup1.RateGroupMemberOut[4] -> ComCcsds.aggregator.timeout

      # Rate group 2
      rateGroupDriver.CycleOut[Ports_RateGroups.rateGroup2] -> rateGroup2.CycleIn
      rateGroup2.RateGroupMemberOut[0] -> cmdSeq.schedIn

      # Rate group 3
      rateGroupDriver.CycleOut[Ports_RateGroups.rateGroup3] -> rateGroup3.CycleIn
      rateGroup3.RateGroupMemberOut[0] -> CdhCore.$health.Run
      rateGroup3.RateGroupMemberOut[1] -> ComCcsds.commsBufferManager.schedIn
      rateGroup3.RateGroupMemberOut[2] -> DataProducts.dpBufferManager.schedIn
      rateGroup3.RateGroupMemberOut[3] -> DataProducts.dpWriter.schedIn
      rateGroup3.RateGroupMemberOut[4] -> DataProducts.dpMgr.schedIn
      # Onboard-table coverage-expiry check on the housekeeping rate group.
      rateGroup3.RateGroupMemberOut[5] -> onboardTables.run
    }

    connections CdhCore_cmdSeq {
      # Command Sequencer
      cmdSeq.comCmdOut -> CdhCore.cmdDisp.seqCmdBuff
      CdhCore.cmdDisp.seqCmdStatus -> cmdSeq.cmdResponseIn
    }

    connections PolarisFsw {
      # GNC reference queries: the attitude estimator reads the Sun ephemeris and
      # EOP it needs for its inertial references off the onboard tables (§8.1,
      # §11.3). Both are synchronous, lock-free point queries.
      attitudeEstimator.getBodyPosition -> onboardTables.getBodyPosition
      attitudeEstimator.getEopAt        -> onboardTables.getEopAt
      # The orbit estimator reads EOP for the ECEF->ECI ingest reduction and
      # the force model's Earth orientation (§8.3).
      orbitEstimator.getEopAt           -> onboardTables.getEopAt

      # The §8.3 orbit solution to the attitude estimator: the position its
      # magnetic and sun references are evaluated at, published earlier in the
      # same rate-group cycle. Through an outage the estimator coasts on this
      # rather than losing the magnetic pair on the next cycle.
      orbitEstimator.orbitStateOut        -> attitudeEstimator.orbitStateIn

      # The §8.0 estimate to its consumer, and the §7 duty-cycle schedule back:
      # a cycle within the rate group, closed deliberately. The estimator runs
      # first and reads the schedule the controller published *last* cycle, which
      # is the period this cycle's magnetometer sample was taken in — the correct
      # pairing, not a staleness bug (see GncPorts.MtqActuation).
      attitudeEstimator.estimateOut       -> attitudeController.estimateIn
      attitudeController.mtqActuationOut  -> attitudeEstimator.mtqActuationIn
    }

    # ----------------------------------------------------------------------
    # SITL lockstep transport (design doc §2.2, §2.4)
    #
    # The self-contained comm stack (TcpClient/ComStub/FrameAccumulator/deframer
    # -> SitlBridge -> framer -> ...) and the barrier-driven rate group are all
    # internal to the imported PolarisSitl.Subtopology. Only one connection
    # crosses the boundary: SitlBridge publishes each STEP epoch to sitlTime,
    # the deployment-wide time source, which lives here rather than in the
    # subtopology (§3.2). A hardware build drops the block below with the import.
    # ----------------------------------------------------------------------
    connections Sitl {
      PolarisSitl.sitlBridge.timeSetOut -> sitlTime.timeSetIn

      # The barrier-driven 10 Hz GNC cycle (§2.4), in dependency order. Member 0
      # is the orbit estimator: it folds in this step's GNSS fix and publishes
      # the position everything downstream is evaluated at. Member 1 is the
      # attitude estimator, which places its magnetic and sun references on that
      # position and runs on this step's measurements. On hardware the SITL
      # subtopology is dropped and this becomes a wall-clock 10 Hz rate group
      # instead (recipe: PolarisFsw/README.md).
      PolarisSitl.sitlRateGroup.RateGroupMemberOut[0] -> orbitEstimator.run
      PolarisSitl.sitlRateGroup.RateGroupMemberOut[1] -> attitudeEstimator.run
      # Member 2 is the controller: it acts on the solution member 1 just
      # published, in the same cycle, and its commands ride the STEP_REPLY the
      # bridge builds after the group returns (§2.4 step 4).
      PolarisSitl.sitlRateGroup.RateGroupMemberOut[2] -> attitudeController.run
      attitudeController.wheelCmdOut -> PolarisSitl.sitlBridge.wheelCmdIn
      attitudeController.mtqCmdOut   -> PolarisSitl.sitlBridge.mtqCmdIn

      # Sensor measurements, SITL end of the GncPorts seam. One line per installed
      # unit, in the order config/spacecraft/leo_smallsat.yaml declares them —
      # port index is vehicle build order (§19.4), and it is the index the
      # estimator's per-unit parameters (SunAlbedoBoresightsBody) and its FDIR
      # events (ImuUnitExcluded) name. Adding a unit is one line here plus the
      # vehicle config, with no port or component change; SitlBridge skips units
      # nothing is connected to.
      #
      # Two IMUs. That makes the flown branch of the §8.2 vote the *pairwise*
      # one: detect a disagreement, then identify the offender against the MEKF's
      # propagated rate. A third unit would buy a median instead, which needs no
      # external reference — see the vehicle config for the trade.
      PolarisSitl.sitlBridge.imuOut[0]           -> attitudeEstimator.imuIn[0]
      PolarisSitl.sitlBridge.imuOut[1]           -> attitudeEstimator.imuIn[1]
      # Six sun sensors on the six faces: full-sky coverage. The best-available
      # unit is never worse than 54.7 deg (the body diagonal), and the *selected*
      # one is bounded by the 60 deg field edge the sigma budget is derived at —
      # two separate facts, both in the vehicle config.
      PolarisSitl.sitlBridge.sunSensorOut[0]     -> attitudeEstimator.sunSensorIn[0]
      PolarisSitl.sitlBridge.sunSensorOut[1]     -> attitudeEstimator.sunSensorIn[1]
      PolarisSitl.sitlBridge.sunSensorOut[2]     -> attitudeEstimator.sunSensorIn[2]
      PolarisSitl.sitlBridge.sunSensorOut[3]     -> attitudeEstimator.sunSensorIn[3]
      PolarisSitl.sitlBridge.sunSensorOut[4]     -> attitudeEstimator.sunSensorIn[4]
      PolarisSitl.sitlBridge.sunSensorOut[5]     -> attitudeEstimator.sunSensorIn[5]
      # Two magnetometers, voted rather than averaged (§8.2): the field is gated
      # per unit against the onboard IGRF magnitude and a two-unit disagreement is
      # attributed by the modelled field, never split down the middle.
      PolarisSitl.sitlBridge.magnetometerOut[0]  -> attitudeEstimator.magnetometerIn[0]
      PolarisSitl.sitlBridge.magnetometerOut[1]  -> attitudeEstimator.magnetometerIn[1]
      # One GNSS receiver, to the §8.3 orbit estimator — the attitude estimator
      # no longer reads the receiver; it reads the orbit solution.
      PolarisSitl.sitlBridge.gnssOut[0]          -> orbitEstimator.gnssIn[0]
      PolarisSitl.sitlBridge.gnssOut[1]          -> orbitEstimator.gnssIn[1]
      # Two star trackers, king-referenced (§8.2). Unit 0 is st_a, the **king**:
      # its mounting defines the body frame, so it is the one index here that is a
      # vehicle-integration decision rather than a wiring choice — StKingUnit must
      # name it. Unit 1 is stated in the king's frame by the ST_ALIGN_CAL
      # correction before it reaches the filter.
      PolarisSitl.sitlBridge.starTrackerOut[0]   -> attitudeEstimator.starTrackerIn[0]
      PolarisSitl.sitlBridge.starTrackerOut[1]   -> attitudeEstimator.starTrackerIn[1]
      # Four wheel tachometers, to the §8.5 momentum management. Every installed
      # wheel is wired: the stored momentum is a sum over the array, so an
      # unconnected unit does not degrade the answer — it refuses it, which is the
      # designed behaviour and a poor way to discover a missing topology line.
      PolarisSitl.sitlBridge.wheelSpeedOut[0]    -> attitudeController.wheelSpeedIn[0]
      PolarisSitl.sitlBridge.wheelSpeedOut[1]    -> attitudeController.wheelSpeedIn[1]
      PolarisSitl.sitlBridge.wheelSpeedOut[2]    -> attitudeController.wheelSpeedIn[2]
      PolarisSitl.sitlBridge.wheelSpeedOut[3]    -> attitudeController.wheelSpeedIn[3]
    }

  }

}
