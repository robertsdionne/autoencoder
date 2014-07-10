add_executable(bloomfilters voxels.cc)
target_link_libraries(bloomfilters glfw ${GLFW_LIBRARIES})
target_link_libraries(bloomfilters rsd)
