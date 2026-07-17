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

# --- Time: TAI master clock, GPS/TT/TDB scales, leap seconds, UTC derivation --
add_library(polaris_time
  time/leap_seconds.cpp
  time/tdb.cpp
  time/utc.cpp)
target_include_directories(polaris_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_time
  PUBLIC polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::time ALIAS polaris_time)

# --- Ephemeris: onboard Chebyshev evaluation of Sun/Moon/planet positions -----
add_library(polaris_ephemeris
  ephemeris/chebyshev.cpp)
target_include_directories(polaris_ephemeris PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_ephemeris
  PUBLIC Eigen3::Eigen polaris_math polaris_time polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::ephemeris ALIAS polaris_ephemeris)

# --- Frames: the single transform library (ECI<->ECEF, IAU 2006/2000A) + EOP --
# Links ERFA (the BSD translation of the IAU SOFA reference implementation) for
# the reduction itself; REQ-CONV-002 requires one transform library, not ad-hoc
# rotations. ERFA is PUBLIC-free: it is an implementation detail, kept out of the
# public headers, so it links PRIVATE.
add_library(polaris_frames
  frames/eci_ecef.cpp)
target_include_directories(polaris_frames PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_frames
  PUBLIC Eigen3::Eigen polaris_math polaris_time polaris_constants
  PRIVATE ERFA::erfa polaris_flags)
add_library(polaris::frames ALIAS polaris_frames)

# --- State: canonical nav product (EstimatedState / TruthState; header-only) -
add_library(polaris_state INTERFACE)
target_include_directories(polaris_state INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_state INTERFACE
  Eigen3::Eigen polaris_math polaris_time polaris_constants)
add_library(polaris::state ALIAS polaris_state)
