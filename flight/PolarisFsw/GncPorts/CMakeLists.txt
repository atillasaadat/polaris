####
# GncPorts: shared GNC sensor-measurement and estimate port/type definitions
# (design doc §8.0, §8.1, §9.1). Interface-only, like SitlPorts and
# OnboardTablesPorts — no sources.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/GncPorts.fpp"
)
