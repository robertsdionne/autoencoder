#include "blob.hpp"
#include "rectifiedlinearlayer.hpp"

namespace autoencoder {

  template <typename F>
  F RectifiedLinearLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) = std::max(F(0.0), bottom.at(0)->value(i));
    }
    top->at(0)->IsValid();
    return 0.0;
  }

  template <typename F>
  void RectifiedLinearLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i) * (bottom->at(0)->value(i) > F(0.0));
    }
    bottom->at(0)->IsValid();
  }

  template class RectifiedLinearLayer<float>;
  template class RectifiedLinearLayer<double>;

}  // namespace autoencoder
