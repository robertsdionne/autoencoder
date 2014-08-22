#ifndef AUTOENCODER_LAYER_H_
#define AUTOENCODER_LAYER_H_

#include <vector>

#include "values.h"

namespace autoencoder {

  class Layer {
  public:
    virtual ~Layer() = default;

    inline void Forward(const Values &bottom, Values *top) {
      ForwardCpu(bottom, top);
    }

    inline void Backward(const Values &top, Values *bottom) {
      BackwardCpu(top, bottom);
    }

  protected:
    virtual void ForwardCpu(const Values &bottom, Values *top) = 0;

    virtual void ForwardGpu(const Values &bottom, Values *top) {
      ForwardCpu(bottom, top);
    }

    virtual void BackwardCpu(const Values &top, Values *bottom) = 0;

    virtual void BackwardGpu(const Values &top, Values *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_H_
