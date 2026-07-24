# Integration tests (design doc §23.1) — the whole truth sim in one process:
# compiled config, loaded reference data, wired resolvers, composed force model,
# RK89 plant. Unlike unit tests these read the committed fixtures, so they need
# the golden directory.
add_executable(polaris_integration_tests
  sim_runner_test.cpp
  sim_closed_loop_orbit_test.cpp
  sim_sitl_lockstep_test.cpp)

target_compile_definitions(polaris_integration_tests
  PRIVATE POLARIS_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")

target_link_libraries(polaris_integration_tests
  PRIVATE
    polaris::scenario
    polaris::io
    polaris::constants
    polaris::math
    polaris::state
    polaris::time
    polaris_test_flags
    GTest::gtest_main)

gtest_discover_tests(polaris_integration_tests)
