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
        assert(std::isfinite(bottom.at(i)->value(j)));
        assert(std::isfinite(mask.at(i).value(j)));
        assert(std::isfinite(scale));
        assert(std::isfinite(mask.at(i).value(j) * scale));
        assert(std::isfinite(bottom.at(i)->value(j) * scale));
        assert(std::isfinite(bottom.at(i)->value(j) * mask.at(i).value(j)));
        assert(std::isfinite(bottom.at(i)->value(j) * mask.at(i).value(j) * scale));
        top->at(i)->value(j) = bottom.at(i)->value(j) * mask.at(i).value(j) * scale;
      }
      bottom.at(i)->IsValid();
      mask.at(i).IsValid();
      assert(std::isfinite(scale));
      top->at(i)->IsValid();
    }
  }

  void DropoutLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    for (auto i = 0; i < bottom->size(); ++i) {
      // std::cout << "top.at(i) " << top.at(i) << std::endl;
      // std::cout << "top.at(i)->width " << top.at(i)->width << std::endl;
      // std::cout << "top.at(i)->values.values.size() " << top.at(i)->values.values.size() << std::endl;
      // std::cout << "top.at(i)->differences.values.size() " << top.at(i)->differences.values.size() << std::endl;
      // std::cout << "bottom->at(i) " << bottom->at(i) << std::endl;
      // std::cout << "bottom->at(i)->width " << bottom->at(i)->width << std::endl;
      // std::cout << "bottom->at(i)->values.values.size() " << bottom->at(i)->values.values.size() << std::endl;
      // std::cout << "bottom->at(i)->differences.values.size() " << bottom->at(i)->differences.values.size() << std::endl;
      // std::cout << "i, j " << i << ", " << 0 << std::endl;
      for (auto j = 0; j < bottom->at(i)->width; ++j) {
        auto top_diff = top.at(i)->difference(j);
        bottom->at(i)->difference(j) = top_diff * mask.at(i).value(j) * scale;
      }
      bottom->at(i)->IsValid();
    }
  }

}  // namespace autoencoder
