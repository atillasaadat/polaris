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

# GMAT drift check — registered only when the GmatConsole binary is present
# (design doc §23.1/§23.2). GMAT independently recomputes the fixture and the test
# fails on drift; without GMAT (the normal case, every PR) it is simply not added,
# and the committed-fixture comparison above still runs. The work lives in the
# skip-gated pytest test (pure GMAT↔JSON, no C++), driven here so `fprime-util
# check` surfaces it for anyone who has GMAT installed.
find_program(GMAT_CONSOLE NAMES GmatConsole)
if(GMAT_CONSOLE)
  get_filename_component(_polaris_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  add_test(
    NAME golden_regen_drift
    COMMAND ${CMAKE_COMMAND} -E env "GMAT_CONSOLE=${GMAT_CONSOLE}"
            uv run pytest -q "${_polaris_root}/tests/tools/test_gmat_drift.py"
    WORKING_DIRECTORY "${_polaris_root}")
  message(STATUS "GMAT found (${GMAT_CONSOLE}); golden_regen_drift test enabled")
else()
  message(STATUS "GMAT not found; golden_regen_drift skipped (committed fixtures still checked)")
endif()
