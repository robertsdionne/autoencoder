#include "innerproductlayer.hpp"
#include "values.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(const Values &weights, const Values &bias)
  : weights(weights), bias(bias) {}

  void InnerProductLayer::ForwardCpu(
      const std::vector<Values *> &bottom, std::vector<Values *> *top) {
    // *top = weights * bottom + bias;
  }

  void InnerProductLayer::BackwardCpu(
      const std::vector<Values *> &top, std::vector<Values *> *bottom) {
    // *bottom = weights * top;
  }

}  // namespace autoencoder
