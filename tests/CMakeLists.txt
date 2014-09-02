include_directories(../source)

add_executable(alltests
    dataloader_test.cpp
    matrixmath_test.cpp
    rectifiedlinearlayer_test.cpp
    taggedsentence_test.cpp
    values_test.cpp)

add_dependencies(alltests ${BLAS_TARGET})

target_link_libraries(alltests autoencoderlib gflags-static gtest gtest_main ${BLAS_LIBRARY})
