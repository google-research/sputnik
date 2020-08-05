# For use in subdirs.
set(SPUTNIK_SRCS)

set(SPUTNIK_SPMM_TEST_SRCS)
set(SPUTNIK_SPMM_BENCHMARK_SRCS)
add_subdirectory(spmm)

set(SPUTNIK_SDDMM_TEST_SRCS)
set(SPUTNIK_SDDMM_BENCHMARK_SRCS)
add_subdirectory(sddmm)

set(DIR_SRCS)

##
### Find all sources in this directory.
##

# Lib srcs.
file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't want DLMC related code.
file(GLOB TMP dlmc*)
list(APPEND FILTER_SRCS ${TMP})

# Don't need matrix utilities.
file(GLOB TMP matrix_utils*)
list(APPEND FILTER_SRCS ${TMP})

foreach(FILE ${FILTER_SRCS})
  list(REMOVE_ITEM DIR_SRCS ${FILE})
endforeach(FILE)

# Add the sources to the build.
list(APPEND SPUTNIK_SRCS ${DIR_SRCS})

# Set .cu.cc files to be compiled as CUDA.
set(SPUTNIK_CUDA_SRCS ${SPUTNIK_SRCS})
list(FILTER SPUTNIK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${SPUTNIK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Create libsputnik.
add_library(sputnik SHARED ${SPUTNIK_SRCS})
target_link_libraries(sputnik ${SPUTNIK_LIBS})

# Optionally build the test suite.
if (BUILD_TEST)
  # Test sources for all targets.
  file(GLOB SPUTNIK_TEST_SRCS matrix_utils*)

  # SpMM test build.
  list(APPEND SPUTNIK_SPMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_SPMM_TEST_CUDA_SRCS ${SPUTNIK_SPMM_TEST_SRCS})
  list(FILTER SPUTNIK_SPMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SPMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_test ${SPUTNIK_SPMM_TEST_SRCS})
  target_link_libraries(spmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # SDDMM test build.
  list(APPEND SPUTNIK_SDDMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_SDDMM_TEST_CUDA_SRCS ${SPUTNIK_SDDMM_TEST_SRCS})
  list(FILTER SPUTNIK_SDDMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SDDMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_test ${SPUTNIK_SDDMM_TEST_SRCS})
  target_link_libraries(sddmm_test sputnik ${SPUTNIK_TEST_LIBS})  
endif()

# Optionally build the benchmark suite.
if (BUILD_BENCHMARK)
  # Benchmark sources for all targets.
  file(GLOB SPUTNIK_BENCHMARK_SRCS matrix_utils*)

  # SpMM benchmark build.
  list(APPEND SPUTNIK_SPMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_SPMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_benchmark ${SPUTNIK_SPMM_BENCHMARK_SRCS})
  target_link_libraries(spmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # SDDMM benchmark build.
  list(APPEND SPUTNIK_SDDMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_SDDMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_benchmark ${SPUTNIK_SDDMM_BENCHMARK_SRCS})
  target_link_libraries(sddmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})  
endif()
