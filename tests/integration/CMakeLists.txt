# Integration tests (design doc §23.1) — the whole truth sim in one process:
# compiled config, loaded reference data, wired resolvers, composed force model,
# RK89 plant. Unlike unit tests these read the committed fixtures, so they need
# the golden directory.
# Source order is scheduling order, deliberately: gtest registers tests in link
# order, ctest discovers them in registration order, and under equal COST it
# starts them in that order. sitl_attitude_control_test.cpp leads because it
# holds the two longest rows in the repo (the ~450 s-of-sim detumble row and
# the inertial hold) — a parallel ctest that starts the longest job last pays
# for it alone at the end of the run.
add_executable(polaris_integration_tests
  sitl_attitude_control_test.cpp
  sitl_fault_matrix_test.cpp
  sitl_attitude_tuning_test.cpp
  sim_sitl_lockstep_test.cpp
  sim_closed_loop_orbit_test.cpp
  sim_runner_test.cpp)

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
# TIMEOUT is sized for CI hardware, not local. The SITL rows integrate whole
# closed-loop mission segments (a detumble row is >200 s of sim time in
# micro-stepped lockstep); on a GitHub runner the long rows measure 300–1500+ s
# against ctest's 1500 s default, which fails the suite on runner speed rather
# than on behaviour.
#
# COST is longest-first scheduling for a parallel ctest. A fresh runner has no
# CostData history, so without it ctest starts tests in definition order and
# the integration rows — the longest in the repo — start last; the makespan is
# then "the whole rest of the suite, then the detumble row alone". Any positive
# value beats the unit suites' implicit 0; the source order above settles the
# tie among these rows.
gtest_discover_tests(polaris_integration_tests
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  PROPERTIES TIMEOUT 3600 COST 1000)
