####
# OnboardTablesPorts: shared onboard-table query port/type definitions
# (design doc §11.3, §22). Interface-only, like SitlPorts — no sources.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/OnboardTablesPorts.fpp"
)
