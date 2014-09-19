#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual ~RectifiedLinearLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
