#include "blob.hpp"
#include "innerproductlayer.hpp"
#include "matrixmath.hpp"

namespace autoencoder {

  InnerProductLayer::InnerProductLayer(Blob<float> &weights, Blob<float> &bias)
  : weights(weights), bias(bias) {}

  float InnerProductLayer::ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) {
    top->at(0)->IsValid();
    for (auto i = 0; i < weights.height; ++i) {
      top->at(0)->value(i) = bias.value(i);
      for (auto k = 0; k < weights.width; ++k) {
        top->at(0)->value(i) += weights.value(k, i) * bottom.at(0)->value(k);
      }
    }
    // Saxpby(1.0f, bias.values, 0.0f, &top->at(0)->values);
    // Sgemv(1.0f, weights.values, bottom.at(0)->values, 1.0f, &top->at(0)->values);
    if (!top->at(0)->IsFinite()) {
      std::cout << "weights " << weights.values << std::endl << std::endl;
      std::cout << "bottom " << bottom.at(0)->values << std::endl;
      std::cout << "top " << top->at(0)->values << std::endl;
    }
    top->at(0)->IsValid();
    return 0.0f;
  }

  void InnerProductLayer::BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) {
    // dE/dW
    for (auto i = 0; i < weights.height; ++i) {
      for (auto j = 0; j < weights.width; ++j) {
        weights.difference(j, i) = bottom->at(0)->value(j) * top.at(0)->difference(i);
      }
    }
    // Sgemm(1.0f, top.at(0)->differences, bottom->at(0)->values, 1.0f, &weights.differences,
    //     CblasTrans);
    // dE/db
    for (auto i = 0; i < bias.width; ++i) {
      bias.difference(i) = top.at(0)->difference(i);
    }
    // Saxpby(1.0f, top.at(0)->differences, 1.0f, &bias.differences);
    // dE/dx
    for (auto i = 0; i < weights.width; ++i) {
      for (auto k = 0; k < weights.height; ++k) {
        bottom->at(0)->difference(i) += weights.value(i, k) * top.at(0)->difference(k);
      }
    }
    // Sgemv(1.0f, weights.values, top.at(0)->differences, 1.0f, &bottom->at(0)->differences,
    //     CblasTrans);
    weights.IsValid();
    bias.IsValid();
    bottom->at(0)->IsValid();
  }

}  // namespace autoencoder
