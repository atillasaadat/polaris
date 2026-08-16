####
# AttitudeController component (design doc §8.5, §7, §9; REQ-ACTL-001..004).
#
# B-dot detumble, quaternion-error PID pointing with L2/L-infinity wheel
# allocation, momentum management with magnetic desaturation and the §8.5
# disturbance feedforward, and the MTQ/MAG duty-cycle interlock the magnetometer
# consumers gate on. The momentum half lives in its own translation unit, as the
# estimator's sensor voting does. Links the flight-safe GNC control laws (polaris_gnc) and the shared
# port/type modules it speaks over — the GncPorts estimate/interlock seam and the
# SitlPorts actuator-command seam. A flight component: only the destination of
# its commands is SITL today.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeControllerMomentum.cpp"
    DEPENDS
        polaris_gnc
        polaris_math
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_SitlPorts
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
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/AttitudeControllerComponentAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")


####
# Component unit test (design doc §23.1). The control *laws* are pinned in
# tests/unit against lib/gnc, so what is under test here is what the component
# adds: the mode ladder and its refusal paths, the parameter-missing refusal, the
# §7 duty-cycle schedule invariants, the §9 stuck-on monitor's confirmation and
# re-admission edges, and the momentum ladder (desaturation engaging and
# disengaging inside POINT, excluded from DETUMBLE, the ground override, the
# envelope monitor and the refusal on a dead tachometer).
####
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeControllerMomentum.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/AttitudeControllerTester.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/AttitudeControllerTestMain.cpp"
    DEPENDS
        polaris_gnc
        polaris_math
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_SitlPorts
        Svc_Sched
        Fw_Time
    UT_AUTO_HELPERS
)
