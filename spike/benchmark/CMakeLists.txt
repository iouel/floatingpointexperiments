cmake_minimum_required(VERSION 3.20)
project(fp_vs_fx64_spike CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Flags are exactly -O3 -mavx2 -mfma (plus the -std=c++20 CMake contributes for
# CMAKE_CXX_STANDARD, identical across all three targets).  CMAKE_BUILD_TYPE is
# forced empty so CMake adds no optimisation flags of its own, and IPO/LTO is
# disabled.  No -march=native, no -ffast-math.
set(CMAKE_BUILD_TYPE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-O3 -mavx2 -mfma")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# fp_a and fp_b are the same fp_double.cpp object; they differ only in link
# order and in the presence of the layout-padding translation unit.
add_executable(fp_a src/harness.cpp src/fp_double.cpp)
add_executable(fp_b src/padding.cpp src/fp_double.cpp src/harness.cpp)
add_executable(fx64 src/harness.cpp src/fixed_u64.cpp)
