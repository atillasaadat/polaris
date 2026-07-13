# Third-party dependencies, pinned and fetched at configure time (design doc
# §18.5 fixed-size Eigen; §23.1 GoogleTest). Both are marked SYSTEM so their
# headers do not trip Polaris's -Werror warning set.

include(FetchContent)

# --- Eigen (header-only linear algebra) -------------------------------------
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)  # suppress Eigen's own test tree
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SYSTEM
)
FetchContent_MakeAvailable(Eigen3)

# --- GoogleTest -------------------------------------------------------------
# The F´ framework can vendor GoogleTest via its own submodule (when framework UTs
# are enabled), so reuse that target when present to avoid a duplicate-target
# clash; otherwise fetch our pinned copy. The Polaris lib unit tests are plain
# GoogleTest (no F´ needed), so this is gated on POLARIS_BUILD_TESTS alone.
if(POLARIS_BUILD_TESTS)
  if(NOT TARGET gtest_main)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE
      SYSTEM
    )
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)

  # --- nlohmann/json (header-only) ------------------------------------------
  # Reads the versioned GMAT golden fixtures (tests/golden/*.json) in the C++
  # comparison harness (§23.1, REQ-VV-002). Test-only; SYSTEM so its headers are
  # exempt from Polaris's -Werror set.
  if(NOT TARGET nlohmann_json::nlohmann_json)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(
      nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG v3.11.3
      GIT_SHALLOW TRUE
      SYSTEM
    )
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()
