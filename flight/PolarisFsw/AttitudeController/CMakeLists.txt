####
# AttitudeController component (design doc §8.5, §7, §9; REQ-ACTL-001..004).
#
# B-dot detumble, quaternion-error PID pointing with L2/L-infinity wheel
# allocation, and the MTQ/MAG duty-cycle interlock the magnetometer consumers
# gate on. Links the flight-safe GNC control laws (polaris_gnc) and the shared
# port/type modules it speaks over — the GncPorts estimate/interlock seam and the
# SitlPorts actuator-command seam. A flight component: only the destination of
# its commands is SITL today.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.cpp"
    DEPENDS
        polaris_gnc
        polaris_math
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_SitlPorts
        Svc_Sched
        Fw_Time
)

####
# Component unit test (design doc §23.1). The control *laws* are pinned in
# tests/unit against lib/gnc, so what is under test here is what the component
# adds: the mode ladder and its refusal paths, the parameter-missing refusal, the
# §7 duty-cycle schedule invariants, and the §9 stuck-on monitor's confirmation
# and re-admission edges.
####
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/AttitudeController.cpp"
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
