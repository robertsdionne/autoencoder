#ifndef VOXELS_VOXELSRENDERER_H_
#define VOXELS_VOXELSRENDERER_H_

#include <glm/glm.hpp>

#include "buffer.h"
#include "drawable.h"
#include "renderer.h"
#include "program.h"
#include "shader.h"
#include "vertexarray.h"
#include "vertexformat.h"

namespace voxels {

  class VoxelsRenderer : public rsd::Renderer {
  public:
    virtual void Change(int width, int height);

    virtual void Create();

    virtual void Render();

  private:
    rsd::Shader vertex_shader, fragment_shader;
    rsd::Program program;
    rsd::VertexFormat vertex_format;
    rsd::VertexArray vertex_array;
    rsd::Buffer vertex_buffer;
    rsd::Drawable triangle;
    glm::mat4 model_view, projection;
    GLuint texture;
  };

}  // namespace voxels

#endif  // VOXELS_VOXELSRENDERER_H_
