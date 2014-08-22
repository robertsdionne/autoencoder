cmake_minimum_required (VERSION 2.8.12)
project(autoencoder CXX)
include_directories(
    libraries/clBLAS/src
    libraries/clFFT/src/include
    libraries/eigen
    libraries/gtest-1.7.0/include
    libraries/opencl/include)
add_subdirectory(libraries/clBLAS/src)
add_subdirectory(libraries/clFFT/src)
add_subdirectory(libraries/gtest-1.7.0)
if (NOT WIN32)
  add_definitions(-std=c++11)
endif()
add_subdirectory(source)
add_subdirectory(tests)
