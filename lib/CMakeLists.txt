# lib/ — shared C++ (design doc §22.3). Flight paths obey the flight memory/
# exception rules (Golden Rule 6). Include root is lib/, so headers are included
# as "constants/...", "math/...".

# --- Physical-constants registry (header-only) ------------------------------
add_library(polaris_constants INTERFACE)
target_include_directories(polaris_constants INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(polaris::constants ALIAS polaris_constants)

# --- Math: frame tags, typed vectors, quaternion, orbit-relative frames ------
add_library(polaris_math
  math/quaternion.cpp
  math/frame_geometry.cpp)
target_include_directories(polaris_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_math
  PUBLIC Eigen3::Eigen polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::math ALIAS polaris_math)

# --- Time: TAI master clock, GPS/TT scales, leap seconds, UTC derivation -----
add_library(polaris_time
  time/leap_seconds.cpp
  time/utc.cpp)
target_include_directories(polaris_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_time
  PUBLIC polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::time ALIAS polaris_time)

# --- State: canonical nav product (EstimatedState / TruthState; header-only) -
add_library(polaris_state INTERFACE)
target_include_directories(polaris_state INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_state INTERFACE
  Eigen3::Eigen polaris_math polaris_time polaris_constants)
add_library(polaris::state ALIAS polaris_state)
