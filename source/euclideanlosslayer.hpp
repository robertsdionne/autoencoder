#ifndef AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
#define AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F> class Device;

  template <typename F>
  class EuclideanLossLayer : public Layer<F> {
  public:
    EuclideanLossLayer(Device<F> &device);

    virtual ~EuclideanLossLayer() = default;

    F ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) override;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
      return F(0.0);
    }

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {}

  private:
    Device<F> &device;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_EUCLIDEANLOSSLAYER_HPP_
