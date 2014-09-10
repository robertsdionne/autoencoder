#include <iostream>

#include "blob.hpp"
#include "euclideanlosslayer.hpp"
#include "matrixmath.hpp"

namespace autoencoder {

  void EuclideanLossLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    // std::cout << bottom.size() << " " << top->size() << std::endl;
    for (auto i = 0; i < bottom.size(); i += 2) {
      for (auto j = 0; j < bottom.at(i)->width; ++j) {
        top->at(i / 2)->difference(j) = bottom.at(i + 1)->value(j) - bottom.at(i)->value(j);
        top->at(i / 2)->value(j) =
            top->at(i / 2)->difference(j) * top->at(i / 2)->difference(j) / 2.0f;
      }
      top->at(i / 2)->IsValid();
    }
  }

  void EuclideanLossLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      auto sign = i == 0 ? 1.0f : -1.0f;
      Saxpby(sign, top.at(i / 2)->differences, 1.0f, &bottom->at(i)->differences);
      bottom->at(i)->IsValid();
    }
  }

}  // namespace autoencoder
