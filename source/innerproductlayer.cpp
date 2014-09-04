#include "innerproductlayer.hpp"
#include "matrixmath.hpp"
#include "parameters.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(Parameters &weights, Parameters &bias)
  : weights(weights), bias(bias) {}

  void InnerProductLayer::ForwardCpu(
      const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) {
    Saxpby(1.0f, bias.values, 0.0f, &top->at(0)->values);
    Sgemv(1.0f, weights.values, bottom.at(0)->values, 1.0f, &top->at(0)->values);
  }

  void InnerProductLayer::BackwardCpu(
      const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) {
    // dE/dW
    Sgemv(1.0f,
        top.at(0)->differences, bottom->at(0)->values, 0.0f, &weights.differences, CblasTrans);
    // dE/db
    Saxpby(1.0f, top.at(0)->differences, 0.0f, &bias.differences);
    // dE/dx
    Sgemv(1.0f, weights.values, top.at(0)->differences, 0.0f, &bottom->at(0)->differences);
  }

}  // namespace autoencoder
