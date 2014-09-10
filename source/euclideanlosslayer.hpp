#ifndef AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
#define AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class EuclideanLossLayer : public Layer {
  public:
    virtual ~EuclideanLossLayer() = default;

    void ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
