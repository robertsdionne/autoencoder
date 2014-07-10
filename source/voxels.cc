#include <GLFW/glfw3.h>
#include <iostream>

void Resize(GLFWwindow *window, int width, int height);

void Setup();

void Draw();

int main(int argument_count, char *arguments[]) {
  if (!glfwInit()) {
    return 1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow *window = glfwCreateWindow(640, 480, "Bloom Filters", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSetWindowSizeCallback(window, Resize);
  Resize(window, 640, 480);
  Setup();
  while (!glfwWindowShouldClose(window)) {
    Draw();
    glfwSwapBuffers(window);
    glfwPollEvents();
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }
  glfwTerminate();
  return 0;
}

void Resize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void Setup() {
  glClearColor(1.0, 0.0, 0.0, 1.0);
}

void Draw() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
