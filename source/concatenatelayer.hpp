#ifndef AUTOENCODER_CONCATENATELAYER_HPP_
#define AUTOENCODER_CONCATENATELAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class ConcatenateLayer : public Layer {
  public:
    virtual ~ConcatenateLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_CONCATENATELAYER_HPP_
