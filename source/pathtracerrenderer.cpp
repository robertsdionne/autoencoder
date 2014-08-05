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
#ifdef WIN32
    vertex_shader.CreateFromFile(GL_VERTEX_SHADER, "vertex.glsl");
    fragment_shader.CreateFromFile(GL_FRAGMENT_SHADER, "fragment.glsl");
    compute_shader.CreateFromFile(GL_COMPUTE_SHADER, "compute.glsl");
#else
    vertex_shader.CreateFromFile(GL_VERTEX_SHADER, "source/vertex.glsl");
    fragment_shader.CreateFromFile(GL_FRAGMENT_SHADER, "source/fragment.glsl");
    compute_shader.CreateFromFile(GL_COMPUTE_SHADER, "source/compute.glsl");
#endif
    program.Create({&vertex_shader, &fragment_shader});
    program.CompileAndLink();
    compute_program.Create({&compute_shader});
    compute_program.CompileAndLink();
    vertex_format.Create({
      {"vertex_position", GL_FLOAT, 2},
      {"tex_coordinate", GL_FLOAT, 2}
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
    frame += 1;
    previous = now;
    now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - start).count() / 1000.0f;
    auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - previous).count() / 1000.0f;
    average = 0.5f * (1.0f / frame_time) + 0.5f * average;
    if (frame % 100 == 0) {
      std::cout << average << std::endl;
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    compute_program.Use();
    compute_program.Uniformsi({
      {"fragments", 0}
    });
    compute_program.Uniformsf({
      {"time", time}
    });
    compute_program.Uniforms2f({
      {"mouse", mouse.get_cursor_position()}
    });
    glDispatchCompute(1024 / 32, 1024 / 32, 1);
    program.Use();
    vertex_array.Bind();
    glDrawArrays(triangle.element_type, 0, triangle.element_count);
  }

}  // namespace pathtracer
