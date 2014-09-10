#ifndef AUTOENCODER_INNERPRODUCTLAYER_HPP_
#define AUTOENCODER_INNERPRODUCTLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class InnerProductLayer : public Layer {
  public:
    InnerProductLayer(Blob &weights, Blob &bias);

    virtual ~InnerProductLayer() = default;

    void ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;

  private:
    Blob &weights;
    Blob &bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_HPP_
