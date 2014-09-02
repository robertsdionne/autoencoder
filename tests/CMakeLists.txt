include_directories(../source)

add_executable(alltests
    matrixmath_test.cpp
    rectifiedlinearlayer_test.cpp
    taggedsentence_test.cpp
    values_test.cpp)

add_dependencies(alltests OpenBLAS)

target_link_libraries(alltests autoencoderlib gflags-static gtest gtest_main openblas)
