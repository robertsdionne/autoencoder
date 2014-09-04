#ifndef AUTOENCODER_SOFTMAXLAYER_HPP_
#define AUTOENCODER_SOFTMAXLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class SoftmaxLayer : public Layer {
  public:
    virtual ~SoftmaxLayer() = default;

    void ForwardCpu(const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_SOFTMAXLAYER_HPP_
