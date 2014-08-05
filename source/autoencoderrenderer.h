#ifndef AUTOENCODER_AUTOENCODERRENDERER_H_
#define AUTOENCODER_AUTOENCODERRENDERER_H_

#include <chrono>
#include <glm/glm.hpp>

#include "buffer.h"
#include "drawable.h"
#include "mouse.h"
#include "program.h"
#include "renderer.h"
#include "shader.h"
#include "vertexarray.h"
#include "vertexformat.h"

namespace autoencoder {

  class AutoencoderRenderer : public rsd::Renderer {
  public:
    AutoencoderRenderer(rsd::Mouse &mouse);

    virtual ~AutoencoderRenderer();

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
    GLuint weights, samples, output;
    float *weight_data, *sample_data, *output_data;
    std::chrono::high_resolution_clock::time_point start, now, previous;
    float average = 0.0f;
    unsigned long long frame = 0;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_AUTOENCODERRENDERER_H_
