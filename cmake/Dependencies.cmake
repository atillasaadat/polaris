# Third-party dependencies, pinned and fetched at configure time (design doc
# §18.5 fixed-size Eigen; §23.1 GoogleTest; §3.1/REQ-CONV-002 ERFA). All are
# marked SYSTEM so their headers do not trip Polaris's -Werror warning set.

include(FetchContent)

# --- Eigen (header-only linear algebra) -------------------------------------
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
# Eigen adds its own (very large) test tree when BUILD_TESTING is on, so it is
# forced off across the fetch — and then **restored**. BUILD_TESTING is not a
# private Eigen switch: F´ keys `register_fprime_ut` off it, so leaving it forced
# off here silently drops every F´ component unit test in the deployment (which
# it did, undetected, until the first one was written in Push 40).
set(POLARIS_SAVED_BUILD_TESTING "${BUILD_TESTING}")
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SYSTEM
)
FetchContent_MakeAvailable(Eigen3)
if (DEFINED POLARIS_SAVED_BUILD_TESTING)
  set(BUILD_TESTING "${POLARIS_SAVED_BUILD_TESTING}" CACHE BOOL "" FORCE)
else()
  unset(BUILD_TESTING CACHE)  # it was never set; leave it that way
endif()

# --- ERFA (Essential Routines for Fundamental Astronomy) ---------------------
# The IAU 2006/2000A ECI<->ECEF reduction used by lib/frames (REQ-CONV-002).
# NOT test-gated: flight code links it. ERFA is C99, and the routines we call
# (eraC2t06a and its tree) are stack-only — no heap, no exceptions — so they
# satisfy the flight rules (Golden Rule 6).
#
# Upstream ships autotools + meson only (no CMakeLists.txt), so FetchContent
# merely populates the source and we compile it ourselves. Two sources are
# excluded: t_erfa_c.c is upstream's test main(), and erfaversion.c needs the
# autotools-generated <config.h> and only serves version-query functions we
# never call. Everything else needs nothing but libc + libm.
FetchContent_Declare(
  erfa
  GIT_REPOSITORY https://github.com/liberfa/erfa.git
  GIT_TAG v2.0.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(erfa)

file(GLOB _erfa_sources CONFIGURE_DEPENDS "${erfa_SOURCE_DIR}/src/*.c")
list(FILTER _erfa_sources EXCLUDE REGEX "(t_erfa_c|erfaversion)\\.c$")
add_library(erfa STATIC ${_erfa_sources})
target_include_directories(erfa SYSTEM PUBLIC "${erfa_SOURCE_DIR}/src")
set_target_properties(erfa PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(ERFA::erfa ALIAS erfa)
unset(_erfa_sources)

# --- NRLMSIS 2.1 (truth atmospheric density, REQ-SIM-002) --------------------
# Fetched, deliberately NOT vendored into the tree. NRLMSIS is licensed under
# NRL's MSIS(R) Open Source Academic Research License Agreement: research,
# academic and non-profit use only, and it forbids licensing derivative works
# for a fee without NRL's written consent. Polaris's own commercial tier
# (LICENSING.md) is therefore incompatible with shipping it, so MSIS must be
# removed — or NRL consent obtained — before any Polaris commercial license is
# sold. Keeping the source out of git history is what makes that removal a
# deletion of this block rather than a history rewrite. See
# THIRD_PARTY_NOTICES.md and sim/world/nrlmsis.hpp.
#
# Upstream is a plain tarball of Fortran 90 with no build system, so as with
# ERFA we populate the source and compile it ourselves. msis2.1_test.F90 is
# upstream's test program (its own main()) and is excluded.
if(POLARIS_BUILD_NRLMSIS)
  # A pre-fetched tarball (POLARIS_NRLMSIS_TARBALL env var) is tried before the
  # NRL server: map.nrl.navy.mil has real outages and a configure-time download
  # makes every build hostage to them. CI caches the tarball keyed on the SHA256
  # below and points this variable at it, so the server only needs to answer on
  # a cold cache. The hash is verified either way, and the file still never
  # enters git (the license constraint above is about the tree, not a CI cache).
  # A local path and a remote URL cannot share ExternalProject's URL list
  # ("invalid in a list"), so this is a selection, not a fallback chain: the
  # pre-fetched file wins outright when present, and a corrupt cache fails the
  # SHA256 check loudly rather than falling through to the network.
  set(_nrlmsis_url
      https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.1/nrlmsis2.1.tar.gz)
  if(DEFINED ENV{POLARIS_NRLMSIS_TARBALL} AND EXISTS "$ENV{POLARIS_NRLMSIS_TARBALL}")
    set(_nrlmsis_url "$ENV{POLARIS_NRLMSIS_TARBALL}")
  endif()
  FetchContent_Declare(
    nrlmsis
    URL "${_nrlmsis_url}"
    URL_HASH SHA256=41e47b29f795d36a5cc252b2858aa2a384c4a7323ace3d48d3ea2f2b37a1a6a8
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  unset(_nrlmsis_url)
  FetchContent_MakeAvailable(nrlmsis)

  # Module dependencies force a fixed compile order, so the sources are listed
  # explicitly rather than globbed (a GLOB would hand gfortran msis_calc before
  # the modules it uses).
  add_library(nrlmsis STATIC
    "${nrlmsis_SOURCE_DIR}/msis_constants.F90"
    "${nrlmsis_SOURCE_DIR}/msis_utils.F90"
    "${nrlmsis_SOURCE_DIR}/msis_init.F90"
    "${nrlmsis_SOURCE_DIR}/msis_gfn.F90"
    "${nrlmsis_SOURCE_DIR}/msis_tfn.F90"
    "${nrlmsis_SOURCE_DIR}/msis_dfn.F90"
    "${nrlmsis_SOURCE_DIR}/msis_calc.F90"
    "${nrlmsis_SOURCE_DIR}/msis_gtd8d.F90"
    "${CMAKE_SOURCE_DIR}/sim/world/msis_shim.F90")
  set_target_properties(nrlmsis PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nrlmsis_modules")
  # -DDBLE selects the model's double-precision path (msis_constants.F90 sets
  # `rp = 8` under it). Upstream calls double precision unnecessary for most
  # applications, but it is what makes the output "exactly match the expected
  # output in msis2.1_test_ref_dp.txt, regardless of the compiler or compiler
  # settings" (readme.txt) — i.e. it is the difference between a golden test that
  # is reproducible across toolchains and one that is not. It also matches the
  # `double` the C++ side and the integrator use throughout.
  #
  # Upstream's own warning profile is not ours; do not apply Polaris's -Werror.
  target_compile_options(nrlmsis PRIVATE -DDBLE -w)

  # The model reads its ~2.5 MB coefficient file at init. It is data shipped in
  # the tarball, not something we commit, so its path is baked in at configure
  # time and surfaced to C++ as POLARIS_MSIS_PARM_PATH.
  set(POLARIS_MSIS_PARM_PATH "${nrlmsis_SOURCE_DIR}/msis21.parm" CACHE FILEPATH
      "Path to the NRLMSIS 2.1 msis21.parm coefficient file")
endif()

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
