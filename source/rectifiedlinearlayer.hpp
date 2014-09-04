#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual ~RectifiedLinearLayer() = default;

  protected:
    void ForwardCpu(const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
