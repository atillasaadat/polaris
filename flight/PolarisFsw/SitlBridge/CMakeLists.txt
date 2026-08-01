####
# SitlBridge component (design doc §2.2, §2.4).
#
# Flight end of the SITL lockstep transport. Links polaris_sitl for the shared
# wire format and the FSW-side decode/reply logic (lib/sitl), and GncPorts for
# the sensor-measurement seam it republishes each STEP_REQ onto (§8.0).
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/SitlBridge.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/SitlBridge.cpp"
    DEPENDS
        polaris_sitl
        flight_PolarisFsw_SitlPorts
        flight_PolarisFsw_GncPorts
        Svc_Ports_CommsPorts
        Svc_Cycle
)
