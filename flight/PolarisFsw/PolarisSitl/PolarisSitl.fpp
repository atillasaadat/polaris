module PolarisSitl {

    # ----------------------------------------------------------------------
    # SITL lockstep transport subtopology (design doc §2.2, §2.4)
    # ----------------------------------------------------------------------
    #
    # Packages the plant<->FSW SITL link as a self-contained, importable F´
    # subtopology so a hardware build can exclude it wholesale (see
    # flight/PolarisFsw/README.md "Delivering without SITL"). It bundles a
    # dedicated comm stack disjoint from the GDS ComCcsds stack: its own
    # TcpClient + ComStub + FrameAccumulator + FprimeDeframer/FprimeFramer,
    # feeding the SitlBridge payload boundary, plus the barrier-driven
    # PassiveRateGroup. The actuator commander is the real GNC
    # AttitudeController, which lives in the main topology (it ships to hardware);
    # the Phase-4 placeholder ScriptedCmdSource it replaced was deleted in Push 54.
    #
    # All instances are inert unless flight_PolarisFsw is given -s <port>: the
    # TcpClient is never started and the rate group is never cycled, so a
    # SITL-off run behaves exactly as a build without this subtopology at all.
    #
    # NOTE: SitlTime is deliberately NOT part of this subtopology. It is the
    # deployment's single time source (the `time connections` handler, §3.2) and
    # must exist in every build, flight included; only its SITL activation is
    # gated at runtime. It stays in the main topology and is wired to this
    # subtopology's timeSetOut across the boundary.

    instance sitlBridge: flight.SitlBridge \
        base id PolarisSitlConfig.BASE_ID + 0x0000

    instance comDriverSitl: Drv.TcpClient \
        base id PolarisSitlConfig.BASE_ID + 0x1000

    instance comStubSitl: Svc.ComStub \
        base id PolarisSitlConfig.BASE_ID + 0x2000

    instance frameAccumulatorSitl: Svc.FrameAccumulator \
        base id PolarisSitlConfig.BASE_ID + 0x3000

    instance deframerSitl: Svc.FprimeDeframer \
        base id PolarisSitlConfig.BASE_ID + 0x4000

    instance framerSitl: Svc.FprimeFramer \
        base id PolarisSitlConfig.BASE_ID + 0x5000

    instance commsBufferManagerSitl: Svc.BufferManager \
        base id PolarisSitlConfig.BASE_ID + 0x6000

    # Passive rate group cycled ONLY by sitlBridge.sitlCycleOut per STEP, never
    # by the wall-clock rateGroupDriver — so the 10 Hz FSW cycle is barrier-
    # driven and deterministic (design doc §2.4 steps 3-4).
    instance sitlRateGroup: Svc.PassiveRateGroup \
        base id PolarisSitlConfig.BASE_ID + 0x7000

    topology Subtopology {
        instance sitlBridge
        instance comDriverSitl
        instance comStubSitl
        instance frameAccumulatorSitl
        instance deframerSitl
        instance framerSitl
        instance commsBufferManagerSitl
        instance sitlRateGroup

        connections Sitl {
            # --- Barrier-driven rate group (design doc §2.4 steps 3-4) ---
            # sitlBridge drives the SITL rate group synchronously per STEP; its
            # members run in port order. Members 0-2 — the orbit estimator, the
            # attitude estimator and the controller — are left to the importing
            # topology (Top/topology.fpp), together with the controller's command
            # connections back to sitlBridge: all three are flight components that
            # ship to hardware, so they do not belong in the SITL subtopology.
            # The order is load-bearing: estimation runs before anything that
            # acts on the estimate.
            sitlBridge.sitlCycleOut             -> sitlRateGroup.CycleIn

            # --- Buffer allocations (shared SITL pool) ---
            comDriverSitl.allocate                -> commsBufferManagerSitl.bufferGetCallee
            comDriverSitl.deallocate              -> commsBufferManagerSitl.bufferSendIn
            frameAccumulatorSitl.bufferAllocate   -> commsBufferManagerSitl.bufferGetCallee
            frameAccumulatorSitl.bufferDeallocate -> commsBufferManagerSitl.bufferSendIn
            framerSitl.bufferAllocate             -> commsBufferManagerSitl.bufferGetCallee
            framerSitl.bufferDeallocate           -> commsBufferManagerSitl.bufferSendIn

            # --- Uplink: driver -> stub -> accumulator -> deframer -> bridge ---
            comDriverSitl.$recv                -> comStubSitl.drvReceiveIn
            comStubSitl.drvReceiveReturnOut    -> comDriverSitl.recvReturnIn
            comStubSitl.dataOut                -> frameAccumulatorSitl.dataIn
            frameAccumulatorSitl.dataReturnOut -> comStubSitl.dataReturnIn
            frameAccumulatorSitl.dataOut       -> deframerSitl.dataIn
            deframerSitl.dataReturnOut         -> frameAccumulatorSitl.dataReturnIn
            deframerSitl.dataOut               -> sitlBridge.dataIn
            sitlBridge.dataReturnOut           -> deframerSitl.dataReturnIn

            # --- Downlink (reply): bridge -> framer -> stub -> driver ---
            sitlBridge.dataOut          -> framerSitl.dataIn
            framerSitl.dataReturnOut    -> sitlBridge.dataReturnIn
            framerSitl.dataOut          -> comStubSitl.dataIn
            comStubSitl.dataReturnOut   -> framerSitl.dataReturnIn
            comStubSitl.comStatusOut    -> framerSitl.comStatusIn
            comStubSitl.drvSendOut      -> comDriverSitl.$send
            comDriverSitl.ready         -> comStubSitl.drvConnected
        }

        # NOTE: one connection crosses this subtopology's boundary —
        # sitlBridge.timeSetOut -> sitlTime.timeSetIn — but sitlTime lives in the
        # main topology (it is the deployment-wide time source, §3.2), so the
        # importing topology makes that connection with a direct qualified
        # reference (PolarisSitl.sitlBridge.timeSetOut). See Top/topology.fpp.

    } # end Subtopology
} # end PolarisSitl
