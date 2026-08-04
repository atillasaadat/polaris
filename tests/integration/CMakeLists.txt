# Integration tests (design doc §23.1) — the whole truth sim in one process:
# compiled config, loaded reference data, wired resolvers, composed force model,
# RK89 plant. Unlike unit tests these read the committed fixtures, so they need
# the golden directory.
add_executable(polaris_integration_tests
  sim_runner_test.cpp
  sim_closed_loop_orbit_test.cpp
  sim_sitl_lockstep_test.cpp
  sitl_attitude_tuning_test.cpp
  sitl_fault_matrix_test.cpp
  sitl_attitude_control_test.cpp)

target_compile_definitions(polaris_integration_tests
  PRIVATE POLARIS_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")

target_link_libraries(polaris_integration_tests
  PRIVATE
    polaris::scenario
    polaris::io
    polaris::constants
    polaris::ephemeris
    polaris::math
    polaris::state
    polaris::time
    polaris_test_flags
    GTest::gtest_main)

# WORKING_DIRECTORY is load-bearing, not tidiness. The SITL cases resolve the
# flight binary, the config compiler, the interpreter and the topology dictionary
# through **repo-relative** paths, and every one of them is behind a
# `GTEST_SKIP()` when it is not found. Under ctest the default working directory
# is the binary dir, so all seven SITL tests skipped silently — green, and
# testing nothing. Running them from the source root is what makes them execute.
gtest_discover_tests(polaris_integration_tests
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
