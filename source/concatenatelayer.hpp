#ifndef AUTOENCODER_CONCATENATELAYER_HPP_
#define AUTOENCODER_CONCATENATELAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F>
  class ConcatenateLayer : public Layer<F> {
  public:
    virtual ~ConcatenateLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_CONCATENATELAYER_HPP_
