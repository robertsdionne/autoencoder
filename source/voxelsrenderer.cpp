#include <GLFW/glfw3.h>

#include "voxelsrenderer.h"

namespace voxels {

  void VoxelsRenderer::Change(int width, int height) {
    glViewport(0, 0, width, height);
  }

  void VoxelsRenderer::Create() {
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
  }

  void VoxelsRenderer::Render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }

}  // namespace voxels
