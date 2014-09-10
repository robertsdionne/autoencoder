#include "blob.hpp"
#include "rectifiedlinearlayer.hpp"

namespace autoencoder {

  void RectifiedLinearLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) = std::max(0.0f, bottom.at(0)->value(i));
    }
    top->at(0)->IsValid();
  }

  void RectifiedLinearLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i) * (bottom->at(0)->value(i) > 0.0f);
    }
    bottom->at(0)->IsValid();
  }

}  // namespace autoencoder
