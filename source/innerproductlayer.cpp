#include "blob.hpp"
#include "innerproductlayer.hpp"
#include "matrixmath.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(Blob &weights, Blob &bias)
  : weights(weights), bias(bias) {}

  void InnerProductLayer::ForwardCpu(const Blobs &bottom, Blobs *top) {
    Saxpby(1.0f, bias.values, 0.0f, &top->at(0)->values);
    Sgemv(1.0f, weights.values, bottom.at(0)->values, 1.0f, &top->at(0)->values);
  }

  void InnerProductLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    // dE/dW
    Sgemv(1.0f,
        top.at(0)->differences, bottom->at(0)->values, 0.0f, &weights.differences, CblasTrans);
    // dE/db
    Saxpby(1.0f, top.at(0)->differences, 0.0f, &bias.differences);
    // dE/dx
    Sgemv(1.0f, weights.values, top.at(0)->differences, 0.0f, &bottom->at(0)->differences);
  }

}  // namespace autoencoder
