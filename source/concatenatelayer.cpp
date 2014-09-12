#include "blob.hpp"
#include "concatenatelayer.hpp"

namespace autoencoder {

  float ConcatenateLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    auto offset = 0;
    for (auto i = 0; i < bottom.size(); ++i) {
      for (auto j = 0; j < bottom.at(i)->width; ++j) {
        top->at(0)->value(j + offset) = bottom.at(i)->value(j);
      }
      offset += bottom.at(i)->width;
    }
    top->at(0)->IsValid();
    return 0.0f;
  }

  void ConcatenateLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    auto offset = 0;
    for (auto i = 0; i < bottom->size(); ++i) {
      for (auto j = 0; j < bottom->at(i)->width; ++j) {
        bottom->at(i)->difference(j) = top.at(0)->difference(j + offset);
      }
      offset += bottom->at(i)->width;
      bottom->at(i)->IsValid();
    }
  }

}  // namespace autoencoder
