#include <cassert>
#include <random>

#include "blob.hpp"
#include "device.hpp"
#include "dropoutlayer.hpp"

namespace autoencoder {

  template <typename F>
  DropoutLayer<F>::DropoutLayer(Device<F> &device, F p, std::mt19937 &generator)
    : device(device), mask(), p(p), scale(1.0f / p), generator(generator), bernoulli(p) {
    assert(0.0f <= p <= 1.0f);
  }

  template <typename F>
  F DropoutLayer<F>::ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    if (Mode::kTrain == mode) {
      mask.clear();
      for (auto i = 0; i < bottom.size(); ++i) {
        mask.emplace_back(bottom.at(i)->width);
        device.Initialize(mask.at(i));
        device.Bernoulli(generator, p, &mask.at(i));
        device.Multiply(scale, bottom.at(i)->values, mask.at(i), &top->at(i)->values);
      }
    } else {
      for (auto i = 0; i < bottom.size(); ++i) {
        device.Copy(bottom.at(i)->values, &top->at(i)->values);
      }
    }
    return 0.0f;
  }

  template <typename F>
  void DropoutLayer<F>::BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      device.Multiply(scale, top.at(i)->differences, mask.at(i), &bottom->at(i)->differences);
    }
  }

  template class DropoutLayer<float>;
  template class DropoutLayer<double>;

}  // namespace autoencoder
