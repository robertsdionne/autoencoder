#ifndef AUTOENCODER_LAYER_HPP_
#define AUTOENCODER_LAYER_HPP_

#include "blob.hpp"

namespace autoencoder {

  enum class Mode {
    kTrain,
    kTest
  };

  template <typename F>
  class Layer {
  public:
    virtual ~Layer() = default;

    inline F Forward(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
      return ForwardXpu(mode, bottom, top);
    }

    inline void Backward(const Blobs<F> &top, Blobs<F> *bottom) {
      BackwardXpu(top, bottom);
    }

    virtual F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) = 0;

    virtual F ForwardGpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
      return ForwardXpu(mode, bottom, top);
    }

    virtual F ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
      return ForwardCpu(mode, bottom, top);
    }

    virtual void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) = 0;

    virtual void BackwardGpu(const Blobs<F> &top, Blobs<F> *bottom) {
      BackwardXpu(top, bottom);
    }

    virtual void BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_HPP_
