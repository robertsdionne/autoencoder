#include "innerproductlayer.hpp"
#include "matrixmath.hpp"
#include "values.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(const Values &weights, const Values &bias)
  : weights(weights), bias(bias) {}

  void InnerProductLayer::ForwardCpu(
      const std::vector<Values *> &bottom, std::vector<Values *> *top) {
    Saxpby(1.0f, bias, 0.0f, top->at(0));
    Sgemv(1.0f, weights, *bottom.at(0), 1.0f, top->at(0));
  }

  void InnerProductLayer::BackwardCpu(
      const std::vector<Values *> &top, std::vector<Values *> *bottom) {
    // *bottom = weights * top;
  }

}  // namespace autoencoder
