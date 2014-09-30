#ifndef AUTOENCODER_INNERPRODUCTLAYER_HPP_
#define AUTOENCODER_INNERPRODUCTLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F> class Device;

  template <typename F>
  class InnerProductLayer : public Layer<F> {
  public:
    InnerProductLayer(Device<F> &device, Blob<F> &weights, Blob<F> &bias);

    virtual ~InnerProductLayer() = default;

    F ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) override;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
      return F(0.0);
    }

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {}

  private:
    Device<F> &device;
    Blob<F> &weights;
    Blob<F> &bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_HPP_
