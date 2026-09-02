cmake_minimum_required(VERSION 3.20)
project(fixed_vs_fp_spike CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Brief section 6: use exactly these flags.  No -march=native, no -ffast-math,
# no LTO, no per-target optimisation differences.  CMAKE_BUILD_TYPE is left
# empty so that CMake contributes no additional optimisation flags.
set(CMAKE_BUILD_TYPE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-O3 -mavx2 -mfma")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# FP-A and FP-B are the same fp_double.cpp object; they differ only in link
# order and in the presence of the layout-padding translation unit.
add_executable(fp_bench_a src/harness.cpp src/fp_double.cpp)
add_executable(fp_bench_b src/padding.cpp src/fp_double.cpp src/harness.cpp)
add_executable(fx64_bench  src/harness.cpp src/fixed_u64.cpp)
add_executable(fx128_bench src/harness.cpp src/fixed_u128.cpp)
