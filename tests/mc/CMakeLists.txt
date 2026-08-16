# Monte Carlo campaign drivers — measurement instruments, deliberately NOT
# registered with ctest, for the same reason tests/benchmark/ is not: each run
# flies orbits of closed-loop sim against a forked deployment, so the campaign
# costs hours of wall clock and makes no pass/fail claim of its own. The verdict
# is produced downstream by `analysis/detumble`, which reads the JSONL this
# writes; see analysis/detumble/README.md for how to re-run the campaign, and
# `--runs 4 --jobs 4` for the fast smoke case that proves the harness works.
add_executable(polaris_detumble_mc
  detumble_mc.cpp)

target_compile_definitions(polaris_detumble_mc
  PRIVATE POLARIS_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")

target_include_directories(polaris_detumble_mc
  PRIVATE "${CMAKE_SOURCE_DIR}/tests/integration")

target_link_libraries(polaris_detumble_mc
  PRIVATE
    polaris::scenario
    polaris::io
    polaris::constants
    polaris::ephemeris
    polaris::math
    polaris::state
    polaris::time
    polaris_test_flags)

# The 7-day orbit-determination campaign (design doc §8.3, §9.2, §13). Same
# rules as the detumble driver above: a measurement instrument, not registered
# with ctest, with the verdict produced downstream by `analysis/od` from the
# JSONL it writes. Unlike detumble it runs in-process rather than forking the
# deployment, because `gnc::OrbitOd` *is* the flight code and its F´ component
# seam does not exist yet — see the file header.
add_executable(polaris_orbit_od_mc
  orbit_od_mc.cpp)

target_compile_definitions(polaris_orbit_od_mc
  PRIVATE
    POLARIS_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden"
    POLARIS_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")

target_link_libraries(polaris_orbit_od_mc
  PRIVATE
    polaris::gnc
    polaris::scenario
    polaris::world
    polaris::frames
    polaris::random
    polaris::constants
    polaris::math
    polaris::state
    polaris::time
    polaris_test_flags)
