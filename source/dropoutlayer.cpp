#include <cassert>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"

namespace autoencoder {

  DropoutLayer::DropoutLayer(float p, unsigned int random_seed)
    : mask(1), p(p), scale(1.0f / p), generator(random_seed), bernoulli(p) {
    assert(0.0f <= p <= 1.0f);
  }

  void DropoutLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    mask.Reshape(bottom.at(0)->width);
    for (auto i = 0; i < bottom.at(0)->width; ++i) {
      mask.value(i) = bernoulli(generator);
      top->at(0)->value(i) = bottom.at(0)->value(i) * mask.value(i) * scale;
    }
  }

  void DropoutLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < mask.width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i) * mask.value(i) * scale;
    }
  }

}  // namespace autoencoder
