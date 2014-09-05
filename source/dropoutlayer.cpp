#include <cassert>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"

namespace autoencoder {

  DropoutLayer::DropoutLayer(float p, std::mt19937 &generator)
    : mask(), p(p), scale(1.0f / p), generator(generator), bernoulli(p) {
    assert(0.0f <= p <= 1.0f);
  }

  void DropoutLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    mask.clear();
    for (auto i = 0; i < bottom.size(); ++i) {
      mask.emplace_back(bottom.at(i)->width);
      for (auto j = 0; j < bottom.at(i)->width; ++j) {
        mask.at(i).value(j) = bernoulli(generator);
        top->at(i)->value(j) = bottom.at(i)->value(j) * mask.at(i).value(j) * scale;
      }
    }
  }

  void DropoutLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      for (auto j = 0; j < bottom->at(i)->width; ++j) {
        bottom->at(i)->difference(j) = top.at(i)->difference(j) * mask.at(i).value(j) * scale;
      }
    }
  }

}  // namespace autoencoder
