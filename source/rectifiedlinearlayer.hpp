#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F>
  class RectifiedLinearLayer : public Layer<F> {
  public:
    virtual ~RectifiedLinearLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
