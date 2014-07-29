#ifndef PATHTRACER_PATHTRACERRENDERER_H_
#define PATHTRACER_PATHTRACERRENDERER_H_

#include <glm/glm.hpp>

#include "buffer.h"
#include "drawable.h"
#include "mouse.h"
#include "program.h"
#include "renderer.h"
#include "shader.h"
#include "vertexarray.h"
#include "vertexformat.h"

namespace pathtracer {

  class PathTracerRenderer : public rsd::Renderer {
  public:
    PathTracerRenderer(rsd::Mouse &mouse);

    virtual void Change(int width, int height);

    virtual void Create();

    virtual void Render();

  private:
    rsd::Mouse &mouse;
    rsd::Shader vertex_shader, fragment_shader, compute_shader;
    rsd::Program program, compute_program;
    rsd::VertexFormat vertex_format;
    rsd::VertexArray vertex_array;
    rsd::Buffer vertex_buffer;
    rsd::Drawable triangle;
    glm::mat4 model_view, projection;
    GLuint texture;
    unsigned char texture_data[1024] = {};
  };

}  // namespace pathtracer

#endif  // PATHTRACER_PATHTRACERRENDERER_H_
