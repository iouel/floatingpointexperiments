cmake_minimum_required(VERSION 3.20)
project(representation
    DESCRIPTION "Fixed-point vs floating-point accumulator/search benchmarking spikes"
    LANGUAGES CXX)

# This is a thin umbrella that wires up the independent benchmark spikes under
# benchmarks/.  Each spike is a self-contained CMake project (its own
# cmake_minimum_required()/project() call, its own compiler flags) so it can
# still be configured and built on its own from its subdirectory exactly as
# before; add_subdirectory() here just lets `cmake -S . -B build` at the repo
# root build everything in one pass.
#
# The benchmarks intentionally pin exact, non-default compiler flags (see
# each benchmark's CMakeLists.txt) to keep codegen comparable across runs, so
# they do not use CMAKE_BUILD_TYPE and are unaffected by it being unset here.

option(BUILD_ACCUMULATOR_BENCHMARK "Configure the fixed-point vs FP accumulator benchmark" ON)
option(BUILD_SEARCH_BENCHMARK "Configure the Kronecker-sequence search benchmark" ON)

if(BUILD_ACCUMULATOR_BENCHMARK)
    add_subdirectory(benchmarks/accumulator)
endif()

if(BUILD_SEARCH_BENCHMARK)
    add_subdirectory(benchmarks/search)
endif()
