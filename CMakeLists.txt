cmake_minimum_required (VERSION 2.8.7)
project(autoencoder CXX)

if (NOT WIN32)
#  add_definitions(-std=c++11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
endif()

add_subdirectory(libraries/clBLAS/src)
add_subdirectory(libraries/clFFT/src)

#add_subdirectory(libraries/gflags-2.1.1)
#add_subdirectory(libraries/gtest-1.7.0)

# Disable client, kernel tests and tests for clBLAS and clFFT to avoid depending upon Boost.
set(BUILD_CLIENT OFF)
set(BUILD_KTEST OFF)
set(BUILD_TEST OFF)

if (APPLE)
  find_library(ACCELERATE_LIBRARY Accelerate)
  find_library(AUDIO_TOOLBOX_LIBRARY AudioToolbox)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(OPENCL_LIBRARY OpenCL)

  add_custom_target(AccelerateBlas)

  set(BLAS_LIBRARY ${ACCELERATE_LIBRARY})
  set(BLAS_TARGET AccelerateBlas)
  set(BLAS_HEADER <Accelerate/Accelerate.h>)
else (APPLE)
# include(ExternalProject)

# ExternalProject_Add(OpenBLAS
#     PREFIX libraries/OpenBLAS-0.2.11
#     SOURCE_DIR libraries/OpenBLAS-0.2.11
#     CONFIGURE_COMMAND true
#     BINARY_DIR libraries/OpenBLAS-0.2.11
#     BUILD_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} $(MAKE)
#     INSTALL_COMMAND true)

# include_directories(libraries/OpenBLAS-0.2.11)

# set(BLAS_LIBRARY openblas)
# set(BLAS_TARGET OpenBLAS)
# set(BLAS_HEADER <cblas.h>)
endif (APPLE)

#add_definitions(-DBLAS_HEADER=${BLAS_HEADER})

include_directories(
    SYSTEM
    libraries/clBLAS/src
    libraries/clFFT/src/include
    libraries/eigen
    libraries/gflags-2.1.1/include
    libraries/gtest-1.7.0/include
    libraries/opencl/include)

add_subdirectory(source)
#add_subdirectory(tests)
