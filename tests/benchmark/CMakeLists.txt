# Benchmarks — measurement instruments, deliberately NOT registered with ctest:
# they attribute wall-clock cost (no pass/fail claim), and putting a timing
# harness behind a CI gate turns runner-speed variance into flakes.
add_executable(polaris_sim_env_bench
  sim_env_ablation_bench.cpp)

target_compile_definitions(polaris_sim_env_bench
  PRIVATE POLARIS_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")

target_include_directories(polaris_sim_env_bench
  PRIVATE "${CMAKE_SOURCE_DIR}/tests/integration")

target_link_libraries(polaris_sim_env_bench
  PRIVATE
    polaris::scenario
    polaris::io
    polaris::constants
    polaris::math
    polaris::state
    polaris::time
    polaris_test_flags)
