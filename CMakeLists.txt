cmake_minimum_required(VERSION 3.13)

project(SPUTNIK CUDA CXX C)

##
### Build options.
##
option(BUILD_TEST "Build test suite" OFF)
option(BUILD_BENCHMARK "Build benchmark suite" OFF)

##
### Handle Dependencies.
##

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

##
### C++ and CUDA flags.
##

# Requires C++11.
set(CMAKE_CXX_STANDARD 11)

message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CUDA_FLAGS_RELEASE is ${CMAKE_CUDA_FLAGS_RELEASE}")
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -NDEBUG")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")

##
### Project build.
##

include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(sputnik)
