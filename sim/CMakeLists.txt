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
  world/drag.cpp
  world/ephemeris_file.cpp
  world/igrf_file.cpp
  world/magnetic_field.cpp
  world/eop_file.cpp
  world/space_weather_file.cpp)
target_include_directories(polaris_world PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_world
  PUBLIC Eigen3::Eigen polaris_dynamics polaris_math polaris_state polaris_time polaris_constants
         polaris_ephemeris polaris_environment polaris_frames
  PRIVATE polaris_flags)
add_library(polaris::world ALIAS polaris_world)

# NRLMSIS 2.1 is optional (research-only license + Fortran toolchain; see
# cmake/Dependencies.cmake and THIRD_PARTY_NOTICES.md). Without it polaris_world
# still builds and the sim uses the exponential atmosphere. POLARIS_HAS_NRLMSIS
# lets dependents compile the model out entirely.
if(POLARIS_BUILD_NRLMSIS)
  target_sources(polaris_world PRIVATE world/nrlmsis.cpp)
  target_link_libraries(polaris_world PUBLIC nrlmsis polaris_frames)
  target_compile_definitions(polaris_world
    PRIVATE POLARIS_MSIS_PARM_PATH="${POLARIS_MSIS_PARM_PATH}"
    PUBLIC POLARIS_HAS_NRLMSIS=1)
endif()

# --- Sensors: truth-in -> measurement-out sensor models (REQ-SIM-003) --------
# The plant's sensor side: each model turns a truth quantity into what the FSW
# would actually read, through the shared §6.1 error stack and its own seeded
# noise stream. Fault-injection hooks are built in for the FDIR suite.
add_library(polaris_sensors
  sensors/magnetometer.cpp
  sensors/imu.cpp
  sensors/occlusion.cpp
  sensors/star_tracker.cpp
  sensors/sun_sensor.cpp)
target_include_directories(polaris_sensors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_sensors
  PUBLIC Eigen3::Eigen polaris_math polaris_state polaris_time polaris_random polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::sensors ALIAS polaris_sensors)

# --- Actuators: command-in -> actual-out actuator models (REQ-SIM-003) --------
# The plant's actuator side: reaction wheels (torque box, friction, quantization,
# power, static/dynamic imbalance for jitter) and magnetorquers (dipole limit,
# residual + hysteresis). Fault-injection hooks are built in for the FDIR suite.
add_library(polaris_actuators
  actuators/reaction_wheel.cpp
  actuators/magnetorquer.cpp)
target_include_directories(polaris_actuators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_actuators
  PUBLIC Eigen3::Eigen polaris_math
  PRIVATE polaris_flags)
add_library(polaris::actuators ALIAS polaris_actuators)

# --- Scenario: compiled-config -> wired environment -> propagation -----------
# Reads the `sim_setup.json` emitted by tools/configc (design doc §19.3 — the
# sim never re-parses the authoritative YAML) and assembles every enabled
# environment model behind one CompositeForceModel.
add_library(polaris_scenario
  scenario/sim_config.cpp
  scenario/sim_runner.cpp
  scenario/vehicle.cpp)
target_include_directories(polaris_scenario PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_scenario
  PUBLIC polaris_world polaris_dynamics polaris_frames polaris_state polaris_time
         polaris_constants polaris_environment polaris_sensors polaris_actuators
         nlohmann_json::nlohmann_json
  PRIVATE polaris_flags)
add_library(polaris::scenario ALIAS polaris_scenario)

# --- The truth-sim executable ------------------------------------------------
add_executable(polaris_sim main.cpp)
target_link_libraries(polaris_sim PRIVATE polaris_scenario polaris_flags)
