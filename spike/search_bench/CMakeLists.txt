cmake_minimum_required(VERSION 3.20)
project(kronecker_search_bench CXX)

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

# search_fp and search_fp_b are the same search_fp.cpp object; they differ only
# in link order and in the presence of the layout-padding translation unit.
add_executable(search_fp   src/harness.cpp src/search_fp.cpp src/reference.cpp)
add_executable(search_fp_b src/padding.cpp src/search_fp.cpp src/reference.cpp src/harness.cpp)
add_executable(search_fx64 src/harness.cpp src/search_fx64.cpp src/reference.cpp)
