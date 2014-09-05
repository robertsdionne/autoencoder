#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "matrixmath.hpp"

namespace autoencoder {

  void EuclideanLossLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    for (auto i = 0; i < bottom.size(); i += 2) {
      for (auto j = 0; j < bottom.at(i)->width; ++j) {
        top->at(i / 2)->difference(j) = bottom.at(i + 1)->value(j) - bottom.at(i)->value(j);
        top->at(i / 2)->value(j) = top->at(i)->difference(j) * top->at(i)->difference(j) / 2.0f;
      }
    }
  }

  void EuclideanLossLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      auto sign = i == 0 ? -1.0f : 1.0f;
      Saxpby(sign, top.at(i / 2)->differences, 1.0f, &bottom->at(i)->differences);
    }
  }

}  // namespace autoencoder
