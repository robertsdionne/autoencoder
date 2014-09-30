#include <iostream>

#include "blob.hpp"
#include "device.hpp"
#include "euclideanlosslayer.hpp"

namespace autoencoder {

  template <typename F>
  EuclideanLossLayer<F>::EuclideanLossLayer(Device<F> &device) : device(device) {}

  template <typename F>
  F EuclideanLossLayer<F>::ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    auto loss = F(0.0);
    for (auto i = 0; i < bottom.size(); i += 2) {
      device.Axpby(F(1.0), bottom.at(i + 1)->values, F(0.0), &top->at(i / 2)->differences);
      device.Axpby(F(-1.0), bottom.at(i)->values, F(1.0), &top->at(i / 2)->differences);
      device.Square(F(1.0 / 2.0), top->at(i / 2)->differences, &top->at(i / 2)->values);
      loss += device.Sum(top->at(i / 2)->values);
    }
    return loss;
  }

  template <typename F>
  void EuclideanLossLayer<F>::BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      auto sign = i % 2 == 0 ? F(-1.0) : F(1.0);
      device.Axpby(sign, top.at(i / 2)->differences, F(1.0), &bottom->at(i)->differences);
    }
  }

  template class EuclideanLossLayer<float>;
  template class EuclideanLossLayer<double>;

}  // namespace autoencoder
