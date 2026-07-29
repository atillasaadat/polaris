####
# F Prime CMakeLists.txt:
#
# SOURCES: list of source files (to be compiled)
# AUTOCODER_INPUTS: list of files to be passed to the autocoders
# DEPENDS: list of libraries that this module depends on
#
# More information in the F´ CMake API documentation:
# https://fprime.jpl.nasa.gov/latest/docs/reference/api/cmake/API/
#
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/instances.fpp"
        "${CMAKE_CURRENT_LIST_DIR}/topology.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/PolarisFswTopology.cpp"
    DEPENDS
        Fw_Logger
        flight_PolarisFsw_SitlBridge
        flight_PolarisFsw_SitlPorts
        flight_PolarisFsw_SitlTime
        flight_PolarisFsw_ScriptedCmdSource
        Svc_PassiveRateGroup
        Svc_ComStub
        Svc_FrameAccumulator
        Svc_FprimeDeframer
        Svc_FprimeFramer
        Svc_BufferManager
        Drv_TcpClient
        polaris_sitl
)
