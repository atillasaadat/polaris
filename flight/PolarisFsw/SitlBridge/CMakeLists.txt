####
# SitlBridge component (design doc §2.2, §2.4).
#
# Flight end of the SITL lockstep transport. Links polaris_sitl for the shared
# wire format and the FSW-side decode/reply logic (lib/sitl).
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/SitlBridge.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/SitlBridge.cpp"
    DEPENDS
        polaris_sitl
        Svc_Ports_CommsPorts
)
