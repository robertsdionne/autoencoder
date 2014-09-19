#ifndef AUTOENCODER_SOFTMAXLAYER_HPP_
#define AUTOENCODER_SOFTMAXLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F>
  class SoftmaxLayer : public Layer<F> {
  public:
    virtual ~SoftmaxLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_SOFTMAXLAYER_HPP_
