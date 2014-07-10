#ifndef VOXELS_VOXELSRENDERER_H_
#define VOXELS_VOXELSRENDERER_H_

#include "renderer.h"

namespace voxels {

  class VoxelsRenderer : public rsd::Renderer {
  public:
    virtual void Change(int width, int height);

    virtual void Create();

    virtual void Render();
  };

}  // namespace voxels

#endif  // VOXELS_VOXELSRENDERER_H_
