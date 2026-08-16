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
        # SITL lockstep transport, packaged as an importable subtopology (§2.2).
        # A hardware build drops this dep (recipe in PolarisFsw/README.md); the
        # SITL component/Svc/Drv modules it groups come in transitively here.
        flight_PolarisFsw_PolarisSitl
        # SitlTime is the deployment-wide time source (stays in the main
        # topology, not the SITL subtopology — see PolarisSitl.fpp).
        flight_PolarisFsw_SitlTime
        # Directly used by the hand-written topology setup (.cpp): the GDS
        # TcpClient (comDriver), the SITL frame detector / buffer bins config,
        # and the shared SITL wire constants.
        Svc_FrameAccumulator
        Svc_BufferManager
        Drv_TcpClient
        polaris_sitl
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

# Same scoped suppression as the component modules: the generated topology
# setup/teardown hooks carry unused `state` parameters.
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/PolarisFswTopologyAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")
