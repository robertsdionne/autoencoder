#include <algorithm>
#include <cmath>
#include <limits>

#include "blob.hpp"
#include "softmaxlayer.hpp"

namespace autoencoder {

  template <typename F>
  F SoftmaxLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    auto maximum = -std::numeric_limits<F>::infinity();
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      maximum = std::max(maximum, bottom.at(0)->value(i));
    }
    auto sum = std::numeric_limits<F>::epsilon();
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      sum += exp(bottom.at(0)->value(i) - maximum);
    }
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) =
          (exp(bottom.at(0)->value(i) - maximum) + std::numeric_limits<F>::epsilon()) / sum;
    }
    top->at(0)->IsValid();
    return 0.0f;
  }

  template <typename F>
  void SoftmaxLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
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
