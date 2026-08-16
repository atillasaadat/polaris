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
