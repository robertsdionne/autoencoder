add_library(autoencoderlib
    dataloader.cpp
    dumbpartofspeechtagger.cpp
    innerproductlayer.cpp
    matrixmath.cpp
    rectifiedlinearlayer.cpp
    taggedsentence.cpp)

add_executable(autoencoder
    dataloader.cpp
    dumbpartofspeechtagger.cpp
    innerproductlayer.cpp
    main.cpp
    matrixmath.cpp
    rectifiedlinearlayer.cpp
    taggedsentence.cpp)
add_executable(clblas clblas.cpp)
add_executable(clfft clfft.cpp)

add_dependencies(autoencoder ${BLAS_TARGET})

target_link_libraries(autoencoder clBLAS clFFT gflags-static ${BLAS_LIBRARY} ${OPENCL_LIBRARY})
target_link_libraries(clblas clBLAS clFFT gflags-static ${OPENCL_LIBRARY})
target_link_libraries(clfft clBLAS clFFT gflags-static ${OPENCL_LIBRARY})
