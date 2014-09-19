#ifndef AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
#define AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F>
  class EuclideanLossLayer : public Layer<F> {
  public:
    virtual ~EuclideanLossLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
