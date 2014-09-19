#ifndef AUTOENCODER_INNERPRODUCTLAYER_HPP_
#define AUTOENCODER_INNERPRODUCTLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class InnerProductLayer : public Layer {
  public:
    InnerProductLayer(Blob<float> &weights, Blob<float> &bias);

    virtual ~InnerProductLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;

  private:
    Blob<float> &weights;
    Blob<float> &bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_HPP_
