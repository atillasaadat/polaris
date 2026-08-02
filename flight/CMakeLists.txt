# flight/ — F´ flight software (design doc §22.3).
#
# This directory is the project's F´ namespace root: it aggregates the flight
# deployment(s) plus, in later phases, the project-wide reusable components and
# ports (flight/components, flight/ports) that GNC deployments share. For now it
# registers the barebones baseline deployment.
#
# Hard flight constraints (no heap/exceptions after init, fixed-size Eigen,
# double-precision nav/control) live in flight/CLAUDE.md and apply to everything
# under here.

# Project overrides of the F´ default configuration (currently the parameter
# database size, §19.3). Registered before the deployment so the override is in
# place when the deployment's components are configured.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/config")

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/PolarisFsw")
