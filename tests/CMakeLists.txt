include_directories(../source)
add_executable(all matrix_test.cpp rectifiedlinearlayer_test.cpp vector_test.cpp)
target_link_libraries(all autoencoderlib gtest gtest_main)
