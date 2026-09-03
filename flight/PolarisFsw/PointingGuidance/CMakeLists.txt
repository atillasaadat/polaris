####
# PointingGuidance component (design doc §8.4; REQ-AGN-004, REQ-AGN-005,
# REQ-ODP-002).
#
# Align/constrain pointing guidance and the target stores it names. Links
# polaris_gnc (the guidance math, the target catalogue and the propagators) and
# polaris_frames (the ECI<->ECEF reduction an ECEF_TARGET needs). A flight
# component — no lib/sitl dependency.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/PointingGuidance.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/PointingGuidance.cpp"
    DEPENDS
        polaris_gnc
        polaris_frames
        flight_PolarisFsw_GncPorts
        flight_PolarisFsw_OnboardTablesPorts
        Svc_Sched
        Fw_Time
)
# The Polaris strict warning set (root CMakeLists) — see OnboardTables for why
# this is applied explicitly rather than inherited.
if(TARGET "${FPRIME_CURRENT_MODULE}")
    get_target_property(_polaris_module_type "${FPRIME_CURRENT_MODULE}" TYPE)
    if(NOT _polaris_module_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries("${FPRIME_CURRENT_MODULE}" PRIVATE polaris_flags)
    endif()
endif()

set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/PointingGuidanceComponentAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")
