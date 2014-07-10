#include <GLFW/glfw3.h>
#include <iostream>

int main(int argument_count, char *arguments[]) {
  if (!glfwInit()) {
    return 1;
  }
  GLFWwindow *window = glfwCreateWindow(640, 480, "Bloom Filters", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  while (!glfwWindowShouldClose(window)) {
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
