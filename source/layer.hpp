#ifndef AUTOENCODER_LAYER_HPP_
#define AUTOENCODER_LAYER_HPP_

#include "blob.hpp"

namespace autoencoder {

  class Layer {
  public:
    virtual ~Layer() = default;

    inline void Forward(const Blobs &bottom, Blobs *top) {
      ForwardCpu(bottom, top);
    }

    inline void Backward(const Blobs &top, Blobs *bottom) {
      BackwardCpu(top, bottom);
    }

    virtual void ForwardCpu(const Blobs &bottom, Blobs *top) = 0;

    virtual void ForwardGpu(const Blobs &bottom, Blobs *top) {
      ForwardCpu(bottom, top);
    }

    virtual void BackwardCpu(const Blobs &top, Blobs *bottom) = 0;

    virtual void BackwardGpu(const Blobs &top, Blobs *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_HPP_
