#include <GLXW/glxw.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "autoencoderrenderer.h"
#include "glfwapplication.h"
#include "mouse.h"

int main(int argument_count, char *arguments[]) {
  rsd::Mouse mouse;
  autoencoder::AutoencoderRenderer renderer(mouse);
  rsd::GlfwApplication application(
      argument_count, arguments, 1024, 1024, 7, "Bloom Filters", renderer, mouse);
  application.Run();
  return 0;
}
