#ifndef AUTOENCODER_INNERPRODUCTLAYER_HPP_
#define AUTOENCODER_INNERPRODUCTLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F>
  class InnerProductLayer : public Layer<F> {
  public:
    InnerProductLayer(Blob<F> &weights, Blob<F> &bias);

    virtual ~InnerProductLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;

  private:
    Blob<F> &weights;
    Blob<F> &bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_HPP_
