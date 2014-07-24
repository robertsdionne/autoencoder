#include <GLFW/glfw3.h>

#include "checks.h"
#include "voxelsrenderer.h"

namespace voxels {

  void VoxelsRenderer::Change(int width, int height) {
    glViewport(0, 0, width, height);
  }

  void VoxelsRenderer::Create() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    vertex_shader.CreateFromFile(GL_VERTEX_SHADER, u8"source/vertex.glsl");
    fragment_shader.CreateFromFile(GL_FRAGMENT_SHADER, u8"source/fragment.glsl");
    program.Create({&vertex_shader, &fragment_shader});
    program.CompileAndLink();
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
    CHECK_STATE(!glGetError());
  }

  void VoxelsRenderer::Render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    program.Use();
    vertex_array.Bind();
    glDrawArrays(triangle.element_type, 0, triangle.element_count);
  }

}  // namespace voxels
