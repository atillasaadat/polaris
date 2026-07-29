####
# ScriptedCmdSource component (design doc §2.4).
#
# Phase-4 placeholder actuator commander on the SITL rate group. Links
# polaris_sitl for the shared deterministic profile, and SitlPorts for the
# command port/array types.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/ScriptedCmdSource.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/ScriptedCmdSource.cpp"
    DEPENDS
        flight_PolarisFsw_SitlPorts
        polaris_sitl
)
