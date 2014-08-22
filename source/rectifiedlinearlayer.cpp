#include <algorithm>
#include <cassert>

#include "rectifiedlinearlayer.h"
#include "vector.h"

namespace autoencoder {

  void RectifiedLinearLayer::ForwardCpu(const Vector &bottom, Vector *top) {
    assert(bottom.width == top->width);
    for (auto i = 0; i < top->width; ++i) {
      top->operator()(i) = std::max(0.0f, bottom(i));
    }
  }

  void RectifiedLinearLayer::BackwardCpu(const Vector &top, Vector *bottom) {
    assert(top.width == bottom->width);
    for (auto i = 0; i < bottom->width; ++i) {
      bottom->operator()(i) = top(i) > 0.0f;
    }
  }

}  // namespace autoencoder
