####
# AttitudeEstimator component (design doc §8.1, §10; REQ-ADET-002, REQ-ADET-003,
# REQ-ADET-004).
#
# Coarse sun + magnetometer + gyro attitude estimator. Links the flight-safe GNC
# chain (polaris_gnc), the onboard IGRF-14 field and its IAGA reader
# (polaris_environment), the ECI<->ECEF reduction (polaris_frames), and the
# shared port/type modules it speaks over. A flight component — no lib/sitl
# dependency; only the source of its measurements is SITL today.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimator.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimator.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimatorMagCal.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimatorSensors.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimatorStarTracker.cpp"
    DEPENDS
        polaris_gnc
        polaris_environment
        polaris_frames
        polaris_time
        polaris_state
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_OnboardTablesPorts
        Svc_Sched
        Fw_Time
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

# The autocoder's default virtual-hook implementations carry unused parameters
# we cannot edit (the file is generated). The suppression is scoped to that one
# generated file — the hand-written sources keep the full polaris_flags set.
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/AttitudeEstimatorComponentAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")


####
# Component unit test (design doc §23.1). The first F´ GTest harness in the
# deployment: the estimation *algorithm* is pinned in tests/unit against lib/gnc,
# so what is under test here is the component itself — reference assembly through
# the OnboardTables ports, the §9.1 staleness gate, parameter refusal, mode and
# health telemetry, and the acquisition/coast/re-acquisition EVR edges. Runs
# under `fprime-util check` with the rest of the UT tree.
####
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimator.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimator.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimatorMagCal.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimatorSensors.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimatorStarTracker.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/AttitudeEstimatorTester.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/AttitudeEstimatorTestMain.cpp"
    DEPENDS
        polaris_gnc
        polaris_environment
        polaris_frames
        polaris_time
        polaris_state
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_OnboardTablesPorts
        Svc_Sched
        Fw_Time
    UT_AUTO_HELPERS
)

if (BUILD_TESTING AND NOT __FPRIME_NO_UT_GEN__)
    # The verbatim IAGA coefficients the onboard snapshot is loaded from, the same
    # file the deployment and the truth sim read.
    target_compile_definitions("${FPRIME_CURRENT_MODULE}_ut_exe"
        PRIVATE POLARIS_IGRF_COEFFS="${CMAKE_SOURCE_DIR}/tests/golden/igrf14coeffs.txt")
endif()
