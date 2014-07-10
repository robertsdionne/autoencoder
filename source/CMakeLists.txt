add_executable(bloomfilters voxels.cpp voxelsrenderer.cpp $<TARGET_OBJECTS:rsd>)
target_link_libraries(bloomfilters glfw ${GLFW_LIBRARIES})
