include_directories(../source)

add_executable(alltests
    concatenatelayer_test.cpp
    cpudevice_test.cpp
    dataloader_test.cpp
    dropoutlayer_test.cpp
    dumbpartofspeechtagger_test.cpp
    euclideanlosslayer_test.cpp
    innerproductlayer_test.cpp
    lookuptable_test.cpp
    partofspeechsentencelayer_test.cpp
    partofspeechwordlayer_test.cpp
    rectifiedlinearlayer_test.cpp
    softmaxlayer_test.cpp
    taggedsentence_test.cpp
    values_test.cpp)

add_dependencies(alltests ${BLAS_TARGET})

target_link_libraries(alltests autoencoderlib gflags-static gtest gtest_main ${BLAS_LIBRARY})
