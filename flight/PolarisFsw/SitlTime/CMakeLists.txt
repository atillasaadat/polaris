####
# SitlTime component (design doc §2.4, §3.2).
#
# The deployment's single FSW time source: sim time in a SITL run, wall-clock
# time otherwise. Depends on SitlPorts for the SitlTimeSet port.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/SitlTime.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/SitlTime.cpp"
    DEPENDS
        flight_PolarisFsw_SitlPorts
        Fw_Time
)
