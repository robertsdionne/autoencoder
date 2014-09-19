#include "blob.hpp"
#include "concatenatelayer.hpp"

namespace autoencoder {

  template <typename F>
  F ConcatenateLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    auto offset = 0;
    for (auto i = 0; i < bottom.size(); ++i) {
      for (auto j = 0; j < bottom.at(i)->width; ++j) {
        top->at(0)->value(j + offset) = bottom.at(i)->value(j);
      }
      offset += bottom.at(i)->width;
    }
    top->at(0)->IsValid();
    return 0.0f;
  }

  template <typename F>
  void ConcatenateLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    auto offset = 0;
    for (auto i = 0; i < bottom->size(); ++i) {
      for (auto j = 0; j < bottom->at(i)->width; ++j) {
        bottom->at(i)->difference(j) = top.at(0)->difference(j + offset);
      }
      offset += bottom->at(i)->width;
      bottom->at(i)->IsValid();
    }
  }

  template class ConcatenateLayer<float>;
  template class ConcatenateLayer<double>;

}  // namespace autoencoder
