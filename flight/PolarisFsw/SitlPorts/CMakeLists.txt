####
# SitlPorts: shared interface-only module for the SITL rate-group ports and
# array types (design doc §2.4). No sources — just autocoded port/type defs that
# SitlBridge, ScriptedCmdSource, and SitlTime all depend on.
####

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/SitlPorts.fpp"
)
