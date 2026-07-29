#####
# 'PolarisFsw' Deployment:
#
# This registers the 'PolarisFsw' deployment to the build system.
# Custom components that have not been added at the project-level should be added to
# the list below.
#
#####

###
# Topology and Components
###

# Custom components (added before Top/ so the topology can reference them).
# SitlPorts (shared port/type defs) first — the others depend on it.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SitlPorts/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SitlTime/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/ScriptedCmdSource/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SitlBridge/")

# Onboard time/EOP/ephemeris tables (design doc §11.3, §22). A flight component
# (ships to hardware, no SITL dependency); its shared query port/type defs
# (OnboardTablesPorts) are registered first, like SitlPorts.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/OnboardTablesPorts/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/OnboardTables/")

# SITL lockstep transport, packaged as an importable subtopology (design doc
# §2.2). Added after the SITL components it groups and before Top, which imports
# it. A hardware build drops this line (see PolarisFsw/README.md).
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/PolarisSitl/")

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Top/")

register_fprime_deployment(
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Main.cpp"
    DEPENDS
        ${FPRIME_CURRENT_MODULE}_Top
)
