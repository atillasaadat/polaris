####
# OnboardTables component (design doc §11.3, §22; REQ-CDH-002).
#
# Onboard time/EOP/ephemeris table provider. Links polaris_onboard (the
# flight-safe table store) and OnboardTablesPorts (the shared query port/type
# defs). A flight component — no lib/sitl dependency.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/OnboardTables.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/OnboardTables.cpp"
    DEPENDS
        polaris_onboard
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
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/OnboardTablesComponentAc.cpp"
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")
