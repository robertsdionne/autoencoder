#ifndef AUTOENCODER_CONCATENATELAYER_HPP_
#define AUTOENCODER_CONCATENATELAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class ConcatenateLayer : public Layer {
  public:
    virtual ~ConcatenateLayer() = default;

    void ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_CONCATENATELAYER_HPP_
