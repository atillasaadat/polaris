# Unit tests for lib/ (design doc §23.1 — test pyramid base).
add_executable(polaris_unit_tests
  constants_test.cpp
  typed_vector_test.cpp
  quaternion_test.cpp
  frame_geometry_test.cpp
  time_test.cpp
  state_test.cpp)

target_link_libraries(polaris_unit_tests
  PRIVATE
    polaris::math
    polaris::constants
    polaris::time
    polaris::state
    polaris_test_flags
    GTest::gtest_main)

gtest_discover_tests(polaris_unit_tests)
