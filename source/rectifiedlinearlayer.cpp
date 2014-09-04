#include <algorithm>
#include <cassert>

#include "rectifiedlinearlayer.hpp"
#include "parameters.hpp"

namespace autoencoder {

  void RectifiedLinearLayer::ForwardCpu(
      const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) {
    assert(bottom.at(0)->width == top->at(0)->width);
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) = std::max(0.0f, bottom.at(0)->value(i));
    }
  }

  void RectifiedLinearLayer::BackwardCpu(
      const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) {
    assert(top.at(0)->width == bottom->at(0)->width);
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i) * (bottom->at(0)->value(i) > 0.0f);
    }
  }

}  // namespace autoencoder
