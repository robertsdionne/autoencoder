#include <iostream>

#include "blob.hpp"
#include "device.hpp"
#include "euclideanlosslayer.hpp"

namespace autoencoder {

  template <typename F>
  EuclideanLossLayer<F>::EuclideanLossLayer(Device<F> &device) : device(device) {}

  template <typename F>
  F EuclideanLossLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    auto loss = 0.0f;
    for (auto i = 0; i < bottom.size(); i += 2) {
      for (auto j = 0; j < bottom.at(i)->width; ++j) {
        top->at(i / 2)->difference(j) = bottom.at(i + 1)->value(j) - bottom.at(i)->value(j);
        top->at(i / 2)->value(j) =
            top->at(i / 2)->difference(j) * top->at(i / 2)->difference(j) / 2.0f;
        loss += top->at(i / 2)->value(j);
      }
      top->at(i / 2)->IsValid();
    }
    return loss;
  }

  template <typename F>
  void EuclideanLossLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      auto sign = i % 2 == 0 ? F(-1.0) : F(1.0);
      device.Axpby(sign, top.at(i / 2)->differences, F(1.0), &bottom->at(i)->differences);
      bottom->at(i)->IsValid();
    }
  }

  template class EuclideanLossLayer<float>;
  template class EuclideanLossLayer<double>;

}  // namespace autoencoder
