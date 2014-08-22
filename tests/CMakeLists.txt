include_directories(../source)
add_executable(alltests rectifiedlinearlayer_test.cpp vector_test.cpp)
target_link_libraries(alltests autoencoderlib gtest gtest_main)
