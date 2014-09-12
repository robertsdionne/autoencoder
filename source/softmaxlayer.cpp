#include <algorithm>
#include <cmath>
#include <limits>

#include "blob.hpp"
#include "softmaxlayer.hpp"

namespace autoencoder {

  float SoftmaxLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    auto maximum = -std::numeric_limits<float>::infinity();
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      maximum = std::max(maximum, bottom.at(0)->value(i));
    }
    auto sum = std::numeric_limits<float>::epsilon();
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      sum += exp(bottom.at(0)->value(i) - maximum);
    }
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) =
          (exp(bottom.at(0)->value(i) - maximum) + std::numeric_limits<float>::epsilon()) / sum;
    }
    top->at(0)->IsValid();
    return 0.0f;
  }

  void SoftmaxLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = 0.0f;
      for (auto j = 0; j < top.at(0)->width; ++j) {
        bottom->at(0)->difference(i) +=
            top.at(0)->difference(j) * top.at(0)->value(j) * ((i == j) - top.at(0)->value(i));
      }
    }
    bottom->at(0)->IsValid();
  }

}  // namespace autoencoder
