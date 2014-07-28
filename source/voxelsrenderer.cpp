#include <GLFW/glfw3.h>
#include <iostream>

#include "checks.h"
#include "voxelsrenderer.h"

extern char etext, edata, end;

namespace voxels {

  VoxelsRenderer::VoxelsRenderer(rsd::Mouse &mouse) : mouse(mouse) {}

  void VoxelsRenderer::Change(int width, int height) {
    glViewport(0, 0, width, height);
  }

  int key(glm::ivec2 v) {
    return ((v.y & 0x3ff) << 10) + (v.x & 0x3ff);
  }

  int h0(int k) {
    return (k * 2654435769 & 0xffffffff) >> (32 - 10);
  }

  int h1(int k) {
    return (k * 3667205999 & 0xffffffff) >> (32 - 10);
  }

  int h2(int k) {
    return (k * 903125161 & 0xffffffff) >> (32 - 10);
  }

  int h3(int k) {
    return (k * 2594636487 & 0xffffffff) >> (32 - 10);
  }

  int h4(int k) {
    return (k * 2458739852 & 0xffffffff) >> (32 - 10);
  }

  int h5(int k) {
    return (k * 4248052 & 0xffffffff) >> (32 - 10);
  }

  void VoxelsRenderer::Create() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    vertex_shader.CreateFromFile(GL_VERTEX_SHADER, u8"source/vertex.glsl");
    fragment_shader.CreateFromFile(GL_FRAGMENT_SHADER, u8"source/fragment.glsl");
    compute_shader.CreateFromFile(GL_COMPUTE_SHADER, u8"source/compute.glsl");
    program.Create({&vertex_shader, &fragment_shader});
    program.CompileAndLink();
    compute_program.Create({&compute_shader});
    compute_program.CompileAndLink();
    vertex_format.Create({
      {u8"vertex_position", GL_FLOAT, 2}
    });
    triangle.data.insert(triangle.data.end(), {
      -1.0f, -1.0f,
       1.0f, -1.0f,
      -1.0f,  1.0f,
       1.0f,  1.0f
    });
    triangle.element_count = 4;
    triangle.element_type = GL_TRIANGLE_STRIP;
    vertex_buffer.Create(GL_ARRAY_BUFFER);
    vertex_buffer.Data(triangle.data_size(), triangle.data.data(), GL_STATIC_DRAW);
    vertex_array.Create();
    vertex_format.Apply(vertex_array, program);
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_1D, texture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RED, 1024, 0, GL_RED, GL_UNSIGNED_BYTE, texture_data);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    CHECK_STATE(!glGetError());
  }

  void VoxelsRenderer::Render() {
    if (mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1) || mouse.IsButtonDown(GLFW_MOUSE_BUTTON_2)) {
      auto position = mouse.get_cursor_position() * glm::vec2(1, -1);
      texture_data[h0(key(position))] = mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1);
      texture_data[h1(key(position))] = mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1);
      texture_data[h2(key(position))] = mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1);
      texture_data[h3(key(position))] = mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1);
      texture_data[h4(key(position))] = mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1);
      texture_data[h5(key(position))] = mouse.IsButtonDown(GLFW_MOUSE_BUTTON_1);
      glTexImage1D(GL_TEXTURE_1D, 0, GL_RED, 1024, 0, GL_RED, GL_UNSIGNED_BYTE, texture_data);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    program.Use();
    vertex_array.Bind();
    glDrawArrays(triangle.element_type, 0, triangle.element_count);
    compute_program.Use();
    glDispatchCompute(100, 100, 1);
  }

}  // namespace voxels
