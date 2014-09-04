set(AUTOENCODER_SOURCES
    concatenatelayer.cpp
    dataloader.cpp
    dropoutlayer.cpp
    dumbpartofspeechtagger.cpp
    euclideanlosslayer.cpp
    evaluator.cpp
    innerproductlayer.cpp
    matrixmath.cpp
    rectifiedlinearlayer.cpp
    recurrentneuralnetworkpartofspeechtagger.cpp
    softmaxlayer.cpp
    taggedsentence.cpp)

add_library(autoencoderlib ${AUTOENCODER_SOURCES})

add_executable(autoencoder main.cpp ${AUTOENCODER_SOURCES})
add_executable(clblas clblas.cpp)
add_executable(clfft clfft.cpp)

add_dependencies(autoencoder ${BLAS_TARGET})

target_link_libraries(autoencoder clBLAS clFFT gflags-static ${BLAS_LIBRARY} ${OPENCL_LIBRARY})
target_link_libraries(clblas clBLAS clFFT gflags-static ${OPENCL_LIBRARY})
target_link_libraries(clfft clBLAS clFFT gflags-static ${OPENCL_LIBRARY})
