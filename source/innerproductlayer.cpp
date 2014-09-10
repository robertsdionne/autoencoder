#include "blob.hpp"
#include "innerproductlayer.hpp"
#include "matrixmath.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(Blob &weights, Blob &bias)
  : weights(weights), bias(bias) {}

  void InnerProductLayer::ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) {
    top->at(0)->IsValid();
    Saxpby(1.0f, bias.values, 0.0f, &top->at(0)->values);
    Sgemv(1.0f, weights.values, bottom.at(0)->values, 1.0f, &top->at(0)->values);
    if (!top->at(0)->IsFinite()) {
      std::cout << "weights " << weights.values << std::endl << std::endl;
      std::cout << "bottom " << bottom.at(0)->values << std::endl;
      std::cout << "top " << top->at(0)->values << std::endl;
    }
    top->at(0)->IsValid();
  }

  void InnerProductLayer::BackwardCpu(const Blobs &top, Blobs *bottom) {
    // dE/dW
    Sgemm(1.0f, top.at(0)->differences, bottom->at(0)->values, 1.0f, &weights.differences,
        CblasTrans);
    // dE/db
    Saxpby(1.0f, top.at(0)->differences, 1.0f, &bias.differences);
    // dE/dx
    Sgemv(1.0f, weights.values, top.at(0)->differences, 1.0f, &bottom->at(0)->differences,
        CblasTrans);
    weights.IsValid();
    bias.IsValid();
    bottom->at(0)->IsValid();
  }

}  // namespace autoencoder
