#include <GLXW/glxw.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <iostream>

#include "checks.h"
#include "pathtracerrenderer.h"

extern char etext, edata, end;

namespace pathtracer {

  PathTracerRenderer::PathTracerRenderer(rsd::Mouse &mouse) : mouse(mouse) {}

  PathTracerRenderer::~PathTracerRenderer() {
    if (texture_data) {
      delete [] texture_data;
    }
  }

  void PathTracerRenderer::Change(int width, int height) {
    glViewport(0, 0, width, height);
  }

  void PathTracerRenderer::Create() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    vertex_shader.CreateFromFile(GL_VERTEX_SHADER, u8"source/vertex.glsl");
    fragment_shader.CreateFromFile(GL_FRAGMENT_SHADER, u8"source/fragment.glsl");
    compute_shader.CreateFromFile(GL_COMPUTE_SHADER, u8"source/compute.glsl");
    program.Create({&vertex_shader, &fragment_shader});
    program.CompileAndLink();
    compute_program.Create({&compute_shader});
    compute_program.CompileAndLink();
    vertex_format.Create({
      {u8"vertex_position", GL_FLOAT, 2},
      {u8"tex_coordinate", GL_FLOAT, 2}
    });
    triangle.data.insert(triangle.data.end(), {
    //    x,     y,     u,     v
      -1.0f, -1.0f,  0.0f,  0.0f,
       1.0f, -1.0f,  1.0f,  0.0f,
      -1.0f,  1.0f,  0.0f,  1.0f,
       1.0f,  1.0f,  1.0f,  1.0f
    });
    triangle.element_count = 4;
    triangle.element_type = GL_TRIANGLE_STRIP;
    vertex_buffer.Create(GL_ARRAY_BUFFER);
    vertex_buffer.Data(triangle.data_size(), triangle.data.data(), GL_STATIC_DRAW);
    vertex_array.Create();
    vertex_format.Apply(vertex_array, program);
    texture_data = new float[1024 * 1024 * 4];
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, texture_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    CHECK_STATE(!glGetError());
    start = std::chrono::high_resolution_clock::now();
  }

  void PathTracerRenderer::Render() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - start).count() / 1000.0f;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    compute_program.Use();
    compute_program.Uniformsi({
      {u8"fragments", 0}
    });
    compute_program.Uniformsf({
      {u8"time", time}
    });
    compute_program.Uniforms2f({
      {u8"mouse", mouse.get_cursor_position()}
    });
    glDispatchCompute(1024 / 32, 1024 / 32, 1);
    program.Use();
    vertex_array.Bind();
    glDrawArrays(triangle.element_type, 0, triangle.element_count);
  }

}  // namespace pathtracer
