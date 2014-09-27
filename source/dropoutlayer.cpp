#include <cassert>
#include <random>

#include "blob.hpp"
#include "dropoutlayer.hpp"

namespace autoencoder {

  template <typename F>
  DropoutLayer<F>::DropoutLayer(F p, std::mt19937 &generator)
    : mask(), p(p), scale(1.0f / p), generator(generator), bernoulli(p) {
    assert(0.0f <= p <= 1.0f);
  }

  template <typename F>
  F DropoutLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    if (Mode::kTrain == mode) {
      auto regenerate_mask = 0 == mask.size();
      for (auto i = 0; i < bottom.size(); ++i) {
        if (regenerate_mask) {
          mask.emplace_back(bottom.at(i)->width);
        }
        for (auto j = 0; j < bottom.at(i)->width; ++j) {
          if (regenerate_mask) {
            mask.at(i).value(j) = bernoulli(generator);
          }
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

  template <typename F>
  void DropoutLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      for (auto j = 0; j < bottom->at(i)->width; ++j) {
        auto top_diff = top.at(i)->difference(j);
        bottom->at(i)->difference(j) = top_diff * mask.at(i).value(j) * scale;
      }
      bottom->at(i)->IsValid();
    }
    mask.clear();
  }

  template class DropoutLayer<float>;
  template class DropoutLayer<double>;

}  // namespace autoencoder
