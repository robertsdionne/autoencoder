#include "blob.hpp"
#include "concatenatelayer.hpp"

namespace autoencoder {

  void ConcatenateLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      top->at(0)->value(i) = bottom.at(0)->value(i);
    }
    for (auto i = 0; i < bottom.at(1)->width; ++i) {
      top->at(0)->value(i + bottom.at(0)->width) = bottom.at(1)->value(i);
    }
  }

  void ConcatenateLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i);
    }
    for (auto i = 0; i < bottom->at(1)->width; ++i) {
      bottom->at(1)->difference(i) = top.at(0)->difference(i + bottom->at(0)->width);
    }
  }

}  // namespace autoencoder
