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
# The Polaris strict warning set (root CMakeLists). Applied here explicitly:
# targets registered through add_fprime_subdirectory inherit F´'s defaults,
# not the project's, so without this line the flight code — the strictest
# lane in the repo — would be the only code compiled with no warnings at all.
# Warnings raised by autocoded sources are fixed by adjusting the FPP or the
# hand-written code, never by weakening this flag set.
if(TARGET "${FPRIME_CURRENT_MODULE}")
    get_target_property(_polaris_module_type "${FPRIME_CURRENT_MODULE}" TYPE)
    if(NOT _polaris_module_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries("${FPRIME_CURRENT_MODULE}" PRIVATE polaris_flags)
    endif()
endif()
