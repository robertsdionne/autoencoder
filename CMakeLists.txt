cmake_minimum_required (VERSION 2.8.12)
project(autoencoder CXX)

include(ExternalProject)

add_subdirectory(libraries/clBLAS/src)
add_subdirectory(libraries/clFFT/src)
add_subdirectory(libraries/gflags-2.1.1)
add_subdirectory(libraries/gtest-1.7.0)

# Disable client, kernel tests and tests for clBLAS and clFFT to avoid depending upon Boost.
set(BUILD_CLIENT OFF)
set(BUILD_KTEST OFF)
set(BUILD_TEST OFF)

if (APPLE)
  find_library(ACCELERATE_LIBRARY Accelerate)
  find_library(AUDIO_TOOLBOX_LIBRARY AudioToolbox)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(OPENCL_LIBRARY OpenCL)
  set(BLAS_LIBRARY ${ACCELERATE_LIBRARY})
else(APPLE)
  ExternalProject_Add(OpenBLAS
      PREFIX libraries/OpenBLAS-0.2.11
      SOURCE_DIR libraries/OpenBLAS-0.2.11
      CONFIGURE_COMMAND true
      BINARY_DIR libraries/OpenBLAS-0.2.11
      BUILD_COMMAND $(MAKE)
      INSTALL_COMMAND true)
  set(BLAS_LIBRARY openblas)
endif(APPLE)

include_directories(
    libraries/OpenBLAS-0.2.11
    libraries/clBLAS/src
    libraries/clFFT/src/include
    libraries/eigen
    libraries/gflags-2.1.1/include
    libraries/gtest-1.7.0/include
    libraries/opencl/include)

if (NOT WIN32)
  add_definitions(-std=c++11)
endif()

add_subdirectory(source)
add_subdirectory(tests)
