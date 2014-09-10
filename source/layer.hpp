#ifndef AUTOENCODER_LAYER_HPP_
#define AUTOENCODER_LAYER_HPP_

#include "blob.hpp"

namespace autoencoder {

  class Layer {
  public:
    enum class Mode {
      kTrain,
      kTest
    };

    virtual ~Layer() = default;

    inline void Forward(Mode mode, const Blobs &bottom, Blobs *top) {
      ForwardCpu(mode, bottom, top);
    }

    inline void Backward(const Blobs &top, Blobs *bottom) {
      BackwardCpu(top, bottom);
    }

    virtual void ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) = 0;

    virtual void ForwardGpu(Mode mode, const Blobs &bottom, Blobs *top) {
      ForwardCpu(mode, bottom, top);
    }

    virtual void BackwardCpu(const Blobs &top, Blobs *bottom) = 0;

    virtual void BackwardGpu(const Blobs &top, Blobs *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_HPP_
