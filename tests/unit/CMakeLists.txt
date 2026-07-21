# Unit tests for lib/ (design doc §23.1 — test pyramid base).
add_executable(polaris_unit_tests
  constants_test.cpp
  typed_vector_test.cpp
  quaternion_test.cpp
  frame_geometry_test.cpp
  eop_test.cpp
  eci_ecef_test.cpp
  time_test.cpp
  tdb_test.cpp
  chebyshev_test.cpp
  state_test.cpp
  sim_dynamics_test.cpp
  sim_world_gravity_test.cpp
  sim_world_egm2008_test.cpp
  sim_world_third_body_test.cpp
  sim_world_srp_test.cpp
  sim_world_drag_test.cpp
  sim_world_ephemeris_file_test.cpp
  igrf_test.cpp
  sim_world_magnetic_field_test.cpp)

# The committed DE440 Chebyshev fit (tests/golden/de440_sun_moon.cheb), the
# verbatim IAGA IGRF-14 coefficients, and the pyIGRF14-generated golden values.
target_compile_definitions(polaris_unit_tests
  PRIVATE POLARIS_EPHEMERIS_FIXTURE="${CMAKE_SOURCE_DIR}/tests/golden/de440_sun_moon.cheb"
          POLARIS_IGRF_COEFFS="${CMAKE_SOURCE_DIR}/tests/golden/igrf14coeffs.txt"
          POLARIS_IGRF_REFERENCE="${CMAKE_SOURCE_DIR}/tests/golden/igrf14_reference.csv")

# NRLMSIS tests only exist when the model was fetched and built; the golden case
# reads upstream's own input/reference files straight from the fetched tree, so
# no MSIS-derived data is committed (see the test file header).
if(POLARIS_BUILD_NRLMSIS)
  target_sources(polaris_unit_tests PRIVATE sim_world_nrlmsis_test.cpp)
  target_compile_definitions(polaris_unit_tests
    PRIVATE POLARIS_MSIS_SOURCE_DIR="${nrlmsis_SOURCE_DIR}")
endif()

target_link_libraries(polaris_unit_tests
  PRIVATE
    polaris::math
    polaris::constants
    polaris::time
    polaris::ephemeris
    polaris::environment
    polaris::state
    polaris::dynamics
    polaris::world
    polaris::frames
    polaris_test_flags
    GTest::gtest_main)

gtest_discover_tests(polaris_unit_tests)
