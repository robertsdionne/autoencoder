if (APPLE)
  find_library(AUDIO_TOOLBOX_LIBRARY AudioToolbox)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(OPENCL_LIBRARY OpenCL)
endif(APPLE)
add_executable(autoencoder autoencoder.cpp main.cpp)
target_link_libraries(
    autoencoder ${AUDIO_TOOLBOX_LIBRARY} ${CORE_FOUNDATION_LIBRARY} ${OPENCL_LIBRARY})
