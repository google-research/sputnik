cmake_minimum_required(VERSION 3.13)

project(SPUTNIK CUDA CXX C)
set(CUDA_ARCHITECTURES OFF)

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

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")

##
### Project build.
##

include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(sputnik)
