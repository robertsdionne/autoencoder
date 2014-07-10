#include <GLFW/glfw3.h>
#include <iostream>

#include "glfwapplication.h"
#include "voxelsrenderer.h"

int main(int argument_count, char *arguments[]) {
  voxels::VoxelsRenderer renderer;
  rsd::GlfwApplication application(argument_count, arguments, 640, 480, "Bloom Filters", renderer);
  application.Run();
  return 0;
}
