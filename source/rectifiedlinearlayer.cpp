#include <cassert>

#include "blob.hpp"
#include "vexcldevice.hpp"
#include "rectifiedlinearlayer.hpp"

namespace autoencoder {

  template <typename F>
  RectifiedLinearLayer<F>::RectifiedLinearLayer(Device<F> &device) : device(device) {}

  template <typename F>
  F RectifiedLinearLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    device.Max(F(0.0), bottom.at(0)->values, &top->at(0)->values);
    top->at(0)->IsValid();
    return F(0.0);
  }

  template <typename F>
  F RectifiedLinearLayer<F>::ForwardGpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    device.Max(F(0.0f), bottom.at(0)->values, &top->at(0)->values);
    return F(0.0);
  }

  template <typename F>
  void RectifiedLinearLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    bottom->at(0)->differences.values =
        top.at(0)->differences.values * (bottom->at(0)->values.values > F(0.0));
    bottom->at(0)->IsValid();
  }

  template class RectifiedLinearLayer<float>;
  template class RectifiedLinearLayer<double>;

}  // namespace autoencoder
