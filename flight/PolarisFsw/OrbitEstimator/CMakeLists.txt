####
# OrbitEstimator component (design doc §8.3, §9.2; REQ-ODP-001, -005, -006).
#
# The F´ seam around the flight-safe polaris::gnc::OrbitOd: latches GNSS fixes
# off the GncPorts measurement seam, runs the position/velocity filter on the
# GNC rate group, reads EOP off the OnboardTables query port, and publishes the
# orbit solution the attitude estimator's references are evaluated at. A flight
# component: only the source of its fixes is SITL today.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/OrbitEstimator.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/OrbitEstimator.cpp"
    DEPENDS
        polaris_gnc
        polaris_frames
        polaris_time
        polaris_constants
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_OnboardTablesPorts
        Svc_Sched
        Fw_Time
)
# The Polaris strict warning set (root CMakeLists): targets registered through
# add_fprime_subdirectory inherit F´'s defaults, not the project's.
if(TARGET "${FPRIME_CURRENT_MODULE}")
    get_target_property(_polaris_module_type "${FPRIME_CURRENT_MODULE}" TYPE)
    if(NOT _polaris_module_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries("${FPRIME_CURRENT_MODULE}" PRIVATE polaris_flags)
    endif()
endif()

# The autocoder's default virtual-hook implementations carry unused parameters
# we cannot edit; the suppression is scoped to that one generated file.
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/OrbitEstimatorComponentAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")

####
# Component unit test (design doc §23.1). The filter itself is pinned in
# tests/unit and tests/golden against lib/gnc; what is under test here is what
# the component adds: seeding from the port, the propagate-then-ingest cycle,
# the coast-horizon drop and its EVR, the parameter-missing refusal, the EOP
# unavailable path, OD_RESET and the published product.
####
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/OrbitEstimator.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/OrbitEstimator.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/OrbitEstimatorTester.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/OrbitEstimatorTestMain.cpp"
    DEPENDS
        polaris_gnc
        polaris_frames
        polaris_time
        polaris_constants
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_OnboardTablesPorts
        Svc_Sched
        Fw_Time
    UT_AUTO_HELPERS
)
