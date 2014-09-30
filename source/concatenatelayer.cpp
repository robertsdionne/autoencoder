#include "blob.hpp"
#include "concatenatelayer.hpp"
#include "device.hpp"

namespace autoencoder {

  template <typename F>
  ConcatenateLayer<F>::ConcatenateLayer(Device<F> &device) : device(device) {}

  template <typename F>
  F ConcatenateLayer<F>::ForwardXpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    auto offset = 0;
    for (auto i = 0; i < bottom.size(); ++i) {
      device.Concatenate(bottom.at(i)->values, offset, &top->at(0)->values);
      offset += bottom.at(i)->width;
    }
    return 0.0f;
  }

  template <typename F>
  void ConcatenateLayer<F>::BackwardXpu(const Blobs<F> &top, Blobs<F> *bottom) {
    auto offset = 0;
    for (auto i = 0; i < bottom->size(); ++i) {
      device.Split(offset, top.at(0)->differences, &bottom->at(i)->differences);
      offset += bottom->at(i)->width;
    }
  }

  template class ConcatenateLayer<float>;
  template class ConcatenateLayer<double>;

}  // namespace autoencoder
