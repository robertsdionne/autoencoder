#include <GLFW/glfw3.h>
#include <iostream>

#include "glfwapplication.h"
#include "voxelsrenderer.h"

int main(int argument_count, char *arguments[]) {
  voxels::VoxelsRenderer renderer;
  rsd::GlfwApplication application(
      argument_count, arguments, 1024, 1024, 7, "Bloom Filters", renderer);
  application.Run();
  return 0;
}
