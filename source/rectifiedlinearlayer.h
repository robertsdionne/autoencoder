#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_H_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_H_

#include "layer.h"
#include "vector.h"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual void ForwardCpu(const Vector &bottom, Vector *top);

    virtual void BackwardCpu(const Vector &top, Vector *bottom);
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_H_
