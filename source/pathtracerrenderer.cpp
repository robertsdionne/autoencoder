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
    if (weight_data) {
      delete [] weight_data;
    }
    if (sample_data) {
      delete [] sample_data;
    }
    if (output_data) {
      delete [] output_data;
    }
  }

  void PathTracerRenderer::Change(int width, int height) {
    glViewport(0, 0, width, height);
  }

  void Print(float *matrix) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 7; ++j) {
        std::cout << matrix[8 * i + j] << " ";
      }
      std::cout << matrix[8 * i + 7] << std::endl;
    }
    std::cout << std::endl;
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
    weight_data = new float[8 * 8];
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        weight_data[8 * i + j] = i == j;
      }
    }
    Print(weight_data);
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &weights);
    glBindTexture(GL_TEXTURE_2D, weights);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 8, 8, 0, GL_RED, GL_FLOAT, weight_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindImageTexture(0, weights, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    sample_data = new float[8 * 8];
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        sample_data[8 * i + j] = 1.0f;
      }
    }
    Print(sample_data);
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &samples);
    glBindTexture(GL_TEXTURE_2D, samples);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 8, 8, 0, GL_RED, GL_FLOAT, sample_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindImageTexture(1, samples, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    output_data = new float[8 * 8];
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        output_data[8 * i + j] = 0.0f;
      }
    }
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &output);
    glBindTexture(GL_TEXTURE_2D, output);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 8, 8, 0, GL_RED, GL_FLOAT, output_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindImageTexture(2, output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    CHECK_STATE(!glGetError());
    start = std::chrono::high_resolution_clock::now();
  }

  void PathTracerRenderer::Render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    compute_program.Use();
    compute_program.Uniformsi({
      {u8"weights", 0},
      {u8"samples", 1},
      {u8"result", 2}
    });
    glDispatchCompute(8 / 2, 8 / 2, 8 / 2);
    glBindTexture(GL_TEXTURE_2D, output);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, output_data);
    Print(output_data);
    exit(0);
  }

}  // namespace pathtracer
