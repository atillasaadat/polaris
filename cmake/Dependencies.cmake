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
if(POLARIS_BUILD_TESTS)
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
  include(GoogleTest)
endif()
