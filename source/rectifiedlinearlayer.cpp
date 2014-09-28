#include <cassert>

#include "blob.hpp"
#include "vexcldevice.hpp"
#include "rectifiedlinearlayer.hpp"

namespace autoencoder {

  template <typename F>
  RectifiedLinearLayer<F>::RectifiedLinearLayer(Device<F> &device) : device(device) {}

  template <typename F>
  F RectifiedLinearLayer<F>::ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    device.Max(F(0.0f), bottom.at(0)->values, &top->at(0)->values);
    return F(0.0);
  }

  template <typename F>
  void RectifiedLinearLayer<F>::BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) {
    device.MaxDerivative(
        F(0.0), top.at(0)->differences, bottom->at(0)->values, &bottom->at(0)->differences);
  }

  template class RectifiedLinearLayer<float>;
  template class RectifiedLinearLayer<double>;

}  // namespace autoencoder
