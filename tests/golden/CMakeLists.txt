# tests/golden/ — GMAT golden-fixture comparison (design doc §23.1, REQ-VV-002).
#
# Numerical Polaris functions are checked against versioned reference fixtures
# (tests/golden/*.json) within documented per-quantity tolerance bands. The
# fixtures are committed data — CI never runs GMAT; GMAT is only used to
# regenerate them (tools/gmat/). GOLDEN_DIR points the tests at this directory
# so they load the checked-in fixtures at runtime.
add_executable(polaris_golden_tests
  gmat_propagation_golden_test.cpp
  orbit_od_golden_test.cpp
  time_scales_golden_test.cpp
  eop_golden_test.cpp
  egm2008_golden_test.cpp)

target_compile_definitions(polaris_golden_tests
  PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(polaris_golden_tests
  PRIVATE
    polaris::time  # transitively brings polaris::constants
    polaris::frames  # ECI<->ECEF reduction + EopTable (eop_golden_test)
    polaris::world  # EGM2008 loader + gravity (egm2008_golden_test), finals.all parser
    polaris::gnc  # onboard OD propagator (orbit_od_golden_test)
    polaris::scenario  # SimRunner, for the GMAT propagation cross-validation
    polaris_test_flags
    nlohmann_json::nlohmann_json
    GTest::gtest_main)

gtest_discover_tests(polaris_golden_tests
  PROPERTIES LABELS golden)

# GMAT drift check — registered only when the GmatConsole binary is present
# (design doc §23.1/§23.2). GMAT independently recomputes the fixture and the test
# fails on drift; without GMAT (the normal case, every PR) it is simply not added,
# and the committed-fixture comparison above still runs. The work lives in the
# skip-gated pytest test (pure GMAT↔JSON, no C++), driven here so `fprime-util
# check` surfaces it for anyone who has GMAT installed.
# GMAT ships the console binary version-suffixed (GmatConsole-R2026a); list the
# bare and current-release names. Setting $GMAT_CONSOLE / running the pytest test
# directly is the version-agnostic path.
find_program(GMAT_CONSOLE NAMES GmatConsole GmatConsole-R2026a
  HINTS ENV GMAT_CONSOLE_DIR "$ENV{HOME}/.cache/polaris-gmat/R2026a/R2026a/bin")
if(GMAT_CONSOLE)
  get_filename_component(_polaris_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  add_test(
    NAME golden_regen_drift
    COMMAND ${CMAKE_COMMAND} -E env "GMAT_CONSOLE=${GMAT_CONSOLE}"
            uv run pytest -q "${_polaris_root}/tests/tools/test_gmat_drift.py"
    WORKING_DIRECTORY "${_polaris_root}")
  set_tests_properties(golden_regen_drift PROPERTIES LABELS golden)
  message(STATUS "GMAT found (${GMAT_CONSOLE}); golden_regen_drift test enabled")
else()
  message(STATUS "GMAT not found; golden_regen_drift skipped (committed fixtures still checked)")
endif()
