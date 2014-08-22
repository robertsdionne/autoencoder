#ifndef AUTOENCODER_LAYER_H_
#define AUTOENCODER_LAYER_H_

#include <vector>

#include "vector.h"

namespace autoencoder {

  class Layer {
  public:
    virtual ~Layer() = default;

    inline void Forward(const Vector &bottom, Vector *top) {
      ForwardCpu(bottom, top);
    }

    inline void Backward(const Vector &top, Vector *bottom) {
      BackwardCpu(top, bottom);
    }

  protected:
    virtual void ForwardCpu(const Vector &bottom, Vector *top) = 0;

    virtual void ForwardGpu(const Vector &bottom, Vector *top) {
      ForwardCpu(bottom, top);
    }

    virtual void BackwardCpu(const Vector &top, Vector *bottom) = 0;

    virtual void BackwardGpu(const Vector &top, Vector *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_H_
