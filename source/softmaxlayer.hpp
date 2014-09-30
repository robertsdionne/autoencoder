#ifndef AUTOENCODER_SOFTMAXLAYER_HPP_
#define AUTOENCODER_SOFTMAXLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F> class Device;

  template <typename F>
  class SoftmaxLayer : public Layer<F> {
  public:
    SoftmaxLayer(Device<F> &device);

    virtual ~SoftmaxLayer() = default;

    F ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) override;

    // TODO(robertsdionne): delete methods below this line.
    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override {
      return F(0.0);
    }

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override {}

  private:
    Device<F> &device;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_SOFTMAXLAYER_HPP_
