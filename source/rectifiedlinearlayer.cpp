#include <algorithm>
#include <cassert>

#include "rectifiedlinearlayer.hpp"
#include "values.hpp"

namespace autoencoder {

  void RectifiedLinearLayer::ForwardCpu(
      const std::vector<Values *> &bottom, std::vector<Values *> *top) {
    assert(bottom.at(0)->width == top->at(0)->width);
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) = std::max(0.0f, bottom.at(0)->value(i));
    }
  }

  void RectifiedLinearLayer::BackwardCpu(
      const std::vector<Values *> &top, std::vector<Values *> *bottom) {
    assert(top.at(0)->width == bottom->at(0)->width);
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i) > 0.0f;
    }
  }

}  // namespace autoencoder
