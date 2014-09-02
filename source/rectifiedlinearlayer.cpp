#include <algorithm>
#include <cassert>

#include "rectifiedlinearlayer.hpp"
#include "values.hpp"

namespace autoencoder {

  void RectifiedLinearLayer::ForwardCpu(const Values &bottom, Values *top) {
    assert(bottom.width == top->width);
    for (auto i = 0; i < top->width; ++i) {
      top->value(i) = std::max(0.0f, bottom.value(i));
    }
  }

  void RectifiedLinearLayer::BackwardCpu(const Values &top, Values *bottom) {
    assert(top.width == bottom->width);
    for (auto i = 0; i < bottom->width; ++i) {
      bottom->difference(i) = top.difference(i) > 0.0f;
    }
  }

}  // namespace autoencoder
