# tests/golden/ — GMAT golden-fixture comparison (design doc §23.1, REQ-VV-002).
#
# Numerical Polaris functions are checked against versioned reference fixtures
# (tests/golden/*.json) within documented per-quantity tolerance bands. The
# fixtures are committed data — CI never runs GMAT; GMAT is only used to
# regenerate them (tools/gmat/). GOLDEN_DIR points the tests at this directory
# so they load the checked-in fixtures at runtime.
add_executable(polaris_golden_tests
  time_scales_golden_test.cpp)

target_compile_definitions(polaris_golden_tests
  PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(polaris_golden_tests
  PRIVATE
    polaris::time  # transitively brings polaris::constants
    polaris_test_flags
    nlohmann_json::nlohmann_json
    GTest::gtest_main)

gtest_discover_tests(polaris_golden_tests)
