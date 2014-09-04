#include <cassert>
#include <cmath>

#include "blob.hpp"
#include "softmaxlayer.hpp"

namespace autoencoder {

  void SoftmaxLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    auto sum = 0.0f;
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      sum += exp(bottom.at(0)->value(i));
    }
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) = exp(bottom.at(0)->value(i)) / sum;
    }
  }

  void SoftmaxLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = 0.0f;
      for (auto j = 0; j < top.at(0)->width; ++j) {
        bottom->at(0)->difference(i) +=
            top.at(0)->difference(j) * top.at(0)->value(j) * ((i == j) - top.at(0)->value(i));
      }
    }
  }

}  // namespace autoencoder
