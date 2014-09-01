include_directories(../source)
add_executable(alltests matrixmath_test.cpp rectifiedlinearlayer_test.cpp values_test.cpp)
target_link_libraries(alltests autoencoderlib gtest gtest_main)
