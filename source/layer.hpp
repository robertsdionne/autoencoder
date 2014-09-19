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

    inline float Forward(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) {
      return ForwardCpu(mode, bottom, top);
    }

    inline void Backward(const Blobs<float> &top, Blobs<float> *bottom) {
      BackwardCpu(top, bottom);
    }

    virtual float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) = 0;

    virtual float ForwardGpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) {
      return ForwardCpu(mode, bottom, top);
    }

    virtual void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) = 0;

    virtual void BackwardGpu(const Blobs<float> &top, Blobs<float> *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_HPP_
