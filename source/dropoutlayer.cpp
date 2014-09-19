#include <cassert>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"

namespace autoencoder {

  DropoutLayer::DropoutLayer(float p, std::mt19937 &generator)
    : mask(), p(p), scale(1.0f / p), generator(generator), bernoulli(p) {
    assert(0.0f <= p <= 1.0f);
  }

  float DropoutLayer::ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) {
    if (Mode::kTrain == mode) {
      mask.clear();
      for (auto i = 0; i < bottom.size(); ++i) {
        mask.emplace_back(bottom.at(i)->width);
        for (auto j = 0; j < bottom.at(i)->width; ++j) {
          mask.at(i).value(j) = bernoulli(generator);
          top->at(i)->value(j) = bottom.at(i)->value(j) * mask.at(i).value(j) * scale;
        }
        bottom.at(i)->IsValid();
        mask.at(i).IsValid();
        top->at(i)->IsValid();
      }
    } else {
      for (auto i = 0; i < bottom.size(); ++i) {
        for (auto j = 0; j < bottom.at(i)->width; ++j) {
          top->at(i)->value(j) = bottom.at(i)->value(j);
        }
      }
    }
    return 0.0f;
  }

  void DropoutLayer::BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      for (auto j = 0; j < bottom->at(i)->width; ++j) {
        auto top_diff = top.at(i)->difference(j);
        bottom->at(i)->difference(j) = top_diff * mask.at(i).value(j) * scale;
      }
      bottom->at(i)->IsValid();
    }
  }

}  // namespace autoencoder
