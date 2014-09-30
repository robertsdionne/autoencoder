#include <algorithm>
#include <cmath>
#include <limits>

#include "blob.hpp"
#include "device.hpp"
#include "softmaxlayer.hpp"

namespace autoencoder {

  template <typename F>
  SoftmaxLayer<F>::SoftmaxLayer(Device<F> &device) : device(device) {}

  template <typename F>
  F SoftmaxLayer<F>::ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    device.Softmax(bottom.at(0)->values, &top->at(0)->values);
    return 0.0f;
  }

  template <typename F>
  void SoftmaxLayer<F>::BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) {
    device.SoftmaxDerivative(
        top.at(0)->values, top.at(0)->differences, &bottom->at(0)->differences);
  }

  template class SoftmaxLayer<float>;
  template class SoftmaxLayer<double>;

}  // namespace autoencoder
