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
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = 0.0f;
      for (auto j = 0; j < top.at(0)->width; ++j) {
        bottom->at(0)->difference(i) +=
            top.at(0)->difference(j) * top.at(0)->value(j) * ((i == j) - top.at(0)->value(i));
      }
    }
    bottom->at(0)->IsValid();
  }

  template class SoftmaxLayer<float>;
  template class SoftmaxLayer<double>;

}  // namespace autoencoder
