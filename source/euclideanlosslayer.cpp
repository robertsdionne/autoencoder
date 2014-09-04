#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "matrixmath.hpp"

namespace autoencoder {

  void EuclideanLossLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      top->at(0)->difference(i) = bottom.at(1)->value(i) - bottom.at(0)->value(i);
      top->at(0)->value(i) = top->at(0)->difference(i) * top->at(0)->difference(i) / 2.0f;
    }
  }

  void EuclideanLossLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      auto sign = i == 0 ? -1.0f : 1.0f;
      Saxpby(sign, top.at(0)->differences, 0.0f, &bottom->at(i)->differences);
    }
  }

}  // namespace autoencoder
