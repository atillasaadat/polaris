# Polaris — top-level build (design doc §23.2). Host targets: WSL/Linux + macOS.
# C++17, warnings-as-errors. Dependencies (Eigen, GoogleTest) are pinned and
# fetched via cmake/Dependencies.cmake.
cmake_minimum_required(VERSION 3.24)

project(Polaris
  VERSION 0.0.0
  DESCRIPTION "Spacecraft GNC flight software & simulation"
  LANGUAGES CXX)

# --- Language standard (design doc §3.6) ------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clang-tidy / editors

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(POLARIS_BUILD_TESTS "Build unit/component tests" ON)
option(POLARIS_WERROR "Treat warnings as errors" ON)
option(POLARIS_SANITIZE "Enable ASan/UBSan (test builds)" OFF)

# --- Common warnings / flags (applied to Polaris targets only) --------------
# JPL-aligned strict warning set (design doc §3.6, §23.2). Third-party headers
# are SYSTEM (see cmake/Dependencies.cmake) so these do not fire on them.
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

include(cmake/Dependencies.cmake)

add_subdirectory(lib)

if(POLARIS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "Polaris ${PROJECT_VERSION}: tests=${POLARIS_BUILD_TESTS} "
               "werror=${POLARIS_WERROR} sanitize=${POLARIS_SANITIZE}")
