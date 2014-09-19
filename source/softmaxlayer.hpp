#ifndef AUTOENCODER_SOFTMAXLAYER_HPP_
#define AUTOENCODER_SOFTMAXLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class SoftmaxLayer : public Layer {
  public:
    virtual ~SoftmaxLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_SOFTMAXLAYER_HPP_
