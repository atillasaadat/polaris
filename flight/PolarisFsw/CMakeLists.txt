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
# Interface-only port/type modules first — the components depend on them.
# SitlPorts: the SITL rate-group/actuator seam (§2.4).
# GncPorts:  the sensor-measurement and estimate seam (§8.0, §8.1) — SitlBridge
#            is the SITL measurement source, so it depends on this too.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SitlPorts/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/GncPorts/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SitlTime/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SitlBridge/")

# Onboard time/EOP/ephemeris tables (design doc §11.3, §22). A flight component
# (ships to hardware, no SITL dependency); its shared query port/type defs
# (OnboardTablesPorts) are registered first, like SitlPorts.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/OnboardTablesPorts/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/OnboardTables/")

# Onboard orbit determination (design doc §8.3, §9.2). A flight component: the
# F´ seam around lib/gnc/orbit_od, fed by the GncPorts GNSS seam and the
# OnboardTables EOP query, and the source of the position the attitude
# estimator's references are evaluated at.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/OrbitEstimator/")
# BurnExecutor: the §17 finite-burn seam (throttles out, commanded acceleration
# to the orbit estimator). Depends on GncPorts and SitlPorts.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/BurnExecutor/")

# Coarse attitude determination (design doc §8.1, §10). A flight component: it
# ships to hardware and consumes the GncPorts measurement seam, whichever source
# fills it (SitlBridge under SITL, Drv sensor drivers on the vehicle).
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/AttitudeEstimator/")

# Attitude control (design doc §8.5, §7). The estimator's consumer and the
# vehicle's actuator commander; it replaced the Phase-4 placeholder
# ScriptedCmdSource in Push 54, which is why that component no longer exists.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/AttitudeController/")

# Align/constrain pointing guidance (design doc §8.4). Sits between the orbit
# estimator (whose state every orbit-relative target is measured from) and the
# controller (which receives only a target attitude and a feedforward rate, and
# deliberately does not know which pointing mode produced them).
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/PointingGuidance/")

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
