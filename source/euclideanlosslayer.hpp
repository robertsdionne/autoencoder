#ifndef AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
#define AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class EuclideanLossLayer : public Layer {
  public:
    virtual ~EuclideanLossLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
