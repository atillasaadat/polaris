####
# PolarisSitl Subtopology (design doc §2.2, §2.4)
#
# The plant<->FSW SITL lockstep transport packaged as an importable F´
# subtopology. INTERFACE module: it contributes FPP (instances + internal
# connections) to the deployment topology, no compiled sources of its own.
#
# To ship a hardware build with zero SITL code, this whole subdirectory is
# dropped and the main topology's `import PolarisSitl.Subtopology` line plus its
# one cross-boundary connection are deleted (recipe in
# flight/PolarisFsw/README.md).
####

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/PolarisSitlConfig/")

register_fprime_module(
    EXCLUDE_FROM_ALL
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/PolarisSitl.fpp"
    INTERFACE
    DEPENDS
        flight_PolarisFsw_PolarisSitl_PolarisSitlConfig
        flight_PolarisFsw_SitlBridge
        flight_PolarisFsw_SitlPorts
        Svc_PassiveRateGroup
        Svc_ComStub
        Svc_FrameAccumulator
        Svc_FprimeDeframer
        Svc_FprimeFramer
        Svc_BufferManager
        Drv_TcpClient
)
