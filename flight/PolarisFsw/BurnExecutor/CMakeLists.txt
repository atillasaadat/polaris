####
# BurnExecutor component (design doc §17, §8.3; REQ-MAN-001).
#
# The finite-burn executor seam: throttles out to the thrusters, the commanded
# non-gravitational acceleration to the orbit filter. The Phase-8 targeting
# commands it; today the SITL rows and the `-b` bench hook do.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/BurnExecutor.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/BurnExecutor.cpp"
    DEPENDS
        polaris_math
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_SitlPorts
        Svc_Sched
        Fw_Time
)
if(TARGET "${FPRIME_CURRENT_MODULE}")
    get_target_property(_polaris_module_type "${FPRIME_CURRENT_MODULE}" TYPE)
    if(NOT _polaris_module_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries("${FPRIME_CURRENT_MODULE}" PRIVATE polaris_flags)
    endif()
endif()
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/BurnExecutorComponentAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")

register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/BurnExecutor.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/BurnExecutor.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/BurnExecutorTester.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/BurnExecutorTestMain.cpp"
    DEPENDS
        polaris_math
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_SitlPorts
        Svc_Sched
        Fw_Time
    UT_AUTO_HELPERS
)
