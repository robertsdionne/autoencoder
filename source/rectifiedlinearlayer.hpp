#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include AUTOENCODER_OPENCL_HEADER

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F> class Device;

  template <typename F>
  class RectifiedLinearLayer : public Layer<F> {
  public:
    RectifiedLinearLayer(Device<F> &device);

    virtual ~RectifiedLinearLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override {
      return F(0.0);
    }

    F ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override {}

    void BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) override;

  private:
    Device<F> &device;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
