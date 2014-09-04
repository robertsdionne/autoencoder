#include <cassert>
#include <gflags/gflags.h>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"

DECLARE_int32(random_seed);

namespace autoencoder {

  DropoutLayer::DropoutLayer(float p)
    : mask(1), p(p), scale(1.0f / p), generator(FLAGS_random_seed), bernoulli(p) {
    assert(0.0f <= p <= 1.0f);
  }

  void DropoutLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    mask.Reshape(bottom.at(0)->width);
    for (auto i = 0; i < mask.width; ++i) {
      mask.value(i) = bernoulli(generator);
      top->at(0)->value(i) = bottom.at(0)->value(i) * mask.value(i) * scale;
    }
  }

  void DropoutLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < mask.width; ++i) {
      bottom->at(0)->value(i) = top.at(0)->value(i) * mask.value(i) * scale;
    }
  }

}  // namespace autoencoder
