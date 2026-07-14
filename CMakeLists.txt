####
# Polaris — top-level build (design doc §23.2).  Host targets: WSL/Linux + macOS.
#
# This is an F´ (F Prime) project: the F´ framework owns the build system, and the
# project is driven with `fprime-util` (which sources settings.ini).  Polaris's own
# shared C++ library (lib/) is a plain CMake target linked into both the F´ flight
# deployment and (later) the sim; the GoogleTest unit suite runs under the F´ build.
#
# Build:   uv sync && uv run fprime-util generate && uv run fprime-util build
# Tests:   uv run fprime-util build --ut && uv run fprime-util check
####
cmake_minimum_required(VERSION 3.24.2)
project(Polaris C CXX)

# --- Language standard (design doc §3.6) ------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clang-tidy / editors

option(POLARIS_BUILD_TESTS "Build Polaris lib unit tests under the F´ UT build" ON)
option(POLARIS_WERROR "Treat warnings as errors on Polaris targets" ON)
option(POLARIS_SANITIZE "Enable ASan/UBSan on Polaris test builds" OFF)

# --- Polaris strict warning set (applied to Polaris targets only) -----------
# JPL-aligned (design doc §3.6, §23.2). Third-party + F´ headers are NOT compiled
# with these; only targets that link polaris_flags get them.
add_library(polaris_flags INTERFACE)
target_compile_options(polaris_flags INTERFACE
  -Wall -Wextra -Wpedantic
  -Wshadow -Wconversion -Wsign-conversion -Wdouble-promotion
  -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual
  -Wnon-virtual-dtor -Wnull-dereference -Wformat=2 -Wimplicit-fallthrough)
if(POLARIS_WERROR)
  target_compile_options(polaris_flags INTERFACE -Werror)
endif()
if(POLARIS_SANITIZE)
  target_compile_options(polaris_flags INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(polaris_flags INTERFACE -fsanitize=address,undefined)
endif()

###
# F´ Core Setup — imports all F prime core components and the F´ make-system.
###
include("${CMAKE_CURRENT_LIST_DIR}/fprime/cmake/FPrime.cmake")
# NOTE: register custom targets between these two lines
fprime_setup_included_code()

# --- Third-party deps consumed by lib/ (Eigen; GoogleTest is provided by F´) -
include(cmake/Dependencies.cmake)

# --- Polaris shared library (plain CMake; consumed by F´ components + sim) ---
add_subdirectory(lib)

# --- Truth / environment simulation (plain CMake; Phase 1+, design doc §5) ---
add_subdirectory(sim)

# --- F´ flight deployment(s) + project-wide components ----------------------
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/flight")

# --- Polaris lib unit tests (plain GoogleTest; configured but built on demand) -
# Not a deployment dependency, so `fprime-util build` won't compile them; build
# explicitly or run via `fprime-util check` (ctest). Set -DPOLARIS_BUILD_TESTS=OFF
# for a pure flight build.
if(POLARIS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "Polaris: F´ project build "
               "(tests=${POLARIS_BUILD_TESTS} werror=${POLARIS_WERROR} sanitize=${POLARIS_SANITIZE})")
