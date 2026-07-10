# tests/ — V&V (design doc §23.1). Tests use a slightly relaxed-but-still
# -Werror flag set: the production warning set (-Wconversion, -Wold-style-cast,
# ...) is kept on library code (polaris_flags) but not forced onto GoogleTest
# macro expansions in test translation units.
add_library(polaris_test_flags INTERFACE)
target_compile_options(polaris_test_flags INTERFACE -Wall -Wextra -Wpedantic)
if(POLARIS_WERROR)
  target_compile_options(polaris_test_flags INTERFACE -Werror)
endif()

add_subdirectory(unit)
