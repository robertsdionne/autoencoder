#include "innerproductlayer.hpp"
#include "values.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(const Values &weights, const Values &bias)
  : weights(weights), bias(bias) {}

  void InnerProductLayer::ForwardCpu(const Values &bottom, Values *top) {
    // *top = weights * bottom + bias;
  }

  void InnerProductLayer::BackwardCpu(const Values &top, Values *bottom) {
    // *bottom = weights * top;
  }

}  // namespace autoencoder
