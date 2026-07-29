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
