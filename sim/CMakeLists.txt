# sim/ — truth / environment simulation (design doc §5, §24 Phase 1).
#
# Plant-side C++: heap/exceptions are permitted here (see sim/CLAUDE.md), unlike
# flight/. Targets are plain CMake libraries linked by the sim executable and the
# test suite; like lib/, nothing here is a flight-deployment dependency. Include
# root is sim/, so headers are included as "dynamics/...". lib/ headers
# ("math/...", "state/...") come transitively from the polaris_* libraries.

# --- Dynamics: RK8(9) integrator + 6DOF rigid-body EOM (REQ-SIM-001) ---------
add_library(polaris_dynamics
  dynamics/integrator.cpp)
target_include_directories(polaris_dynamics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_dynamics
  PUBLIC Eigen3::Eigen polaris_math polaris_state polaris_time polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::dynamics ALIAS polaris_dynamics)

# --- World: environment models (gravity, ... ) (REQ-SIM-002) -----------------
add_library(polaris_world
  world/gravity_field.cpp
  world/egm2008.cpp
  world/third_body.cpp
  world/eclipse.cpp
  world/srp.cpp
  world/atmosphere.cpp
  world/drag.cpp)
target_include_directories(polaris_world PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_world
  PUBLIC Eigen3::Eigen polaris_dynamics polaris_math polaris_state polaris_time polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::world ALIAS polaris_world)
