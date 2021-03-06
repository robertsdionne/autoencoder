#include "blob.hpp"
#include "device.hpp"
#include "innerproductlayer.hpp"

namespace autoencoder {

  template <typename F>
  InnerProductLayer<F>::InnerProductLayer(Device<F> &device, Blob<F> &weights, Blob<F> &bias)
  : device(device), weights(weights), bias(bias) {}

  template <typename F>
  F InnerProductLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    // top->at(0)->IsValid();
    // for (auto i = 0; i < weights.height; ++i) {
    //   top->at(0)->value(i) = bias.value(i);
    //   for (auto k = 0; k < weights.width; ++k) {
    //     top->at(0)->value(i) += weights.value(k, i) * bottom.at(0)->value(k);
    //   }
    // }
    device.Axpby(F(1.0), bias.values, 0.0f, &top->at(0)->values);
    device.Gemv(F(1.0), weights.values, bottom.at(0)->values, F(1.0), &top->at(0)->values);
    if (!top->at(0)->IsFinite()) {
      std::cout << "weights " << weights.values << std::endl << std::endl;
      std::cout << "bottom " << bottom.at(0)->values << std::endl;
      std::cout << "top " << top->at(0)->values << std::endl;
    }
    top->at(0)->IsValid();
    return 0.0f;
  }

  template <typename F>
  void InnerProductLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    // dE/dW
    // for (auto i = 0; i < weights.height; ++i) {
    //   for (auto j = 0; j < weights.width; ++j) {
    //     weights.difference(j, i) = bottom->at(0)->value(j) * top.at(0)->difference(i);
    //   }
    // }
    device.Gemm(F(1.0), top.at(0)->differences, bottom->at(0)->values, F(1.0),
        &weights.differences, Transpose::kYes);
    // dE/db
    // for (auto i = 0; i < bias.width; ++i) {
    //   bias.difference(i) = top.at(0)->difference(i);
    // }
    device.Axpby(F(1.0), top.at(0)->differences, F(1.0), &bias.differences);
    // dE/dx
    // for (auto i = 0; i < weights.width; ++i) {
    //   for (auto k = 0; k < weights.height; ++k) {
    //     bottom->at(0)->difference(i) += weights.value(i, k) * top.at(0)->difference(k);
    //   }
    // }
    device.Gemv(F(1.0), weights.values, top.at(0)->differences, F(1.0),
        &bottom->at(0)->differences, Transpose::kYes);
    weights.IsValid();
    bias.IsValid();
    bottom->at(0)->IsValid();
  }

  template class InnerProductLayer<float>;
  template class InnerProductLayer<double>;

}  // namespace autoencoder
