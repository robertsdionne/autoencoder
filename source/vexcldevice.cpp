#include <cassert>
#include <vexcl/vexcl.hpp>

#include "blob.hpp"
#include "values.hpp"
#include "vexcldevice.hpp"

namespace autoencoder {

  template <typename F>
  VexClDevice<F>::VexClDevice() : context{
    vex::Filter::Type{CL_DEVICE_TYPE_GPU} && vex::Filter::DoublePrecision
  } {}

  template <>
  void VexClDevice<float>::Axpby(
      float alpha, const Values<float> &x, float beta, Values<float> *y) {
    // TODO(robertsdionne): implement.
  }

  template <>
  void VexClDevice<double>::Axpby(
      double alpha, const Values<double> &x, double beta, Values<double> *y) {
    // TODO(robertsdionne): implement.
  }

  template <>
  void VexClDevice<float>::Gemm(
      float alpha, const Values<float> &A, const Values<float> &B, float beta, Values<float> *C,
      Transpose transpose_A, Transpose transpose_B) {
    // TODO(robertsdionne): implement.
  }

  template <>
  void VexClDevice<double>::Gemm(
      double alpha, const Values<double> &A, const Values<double> &B,
      double beta, Values<double> *C, Transpose transpose_A, Transpose transpose_B) {
    // TODO(robertsdionne): implement.
  }

  template <>
  void VexClDevice<float>::Gemv(
      float alpha, const Values<float> &A, const Values<float> &x, float beta, Values<float> *y,
      Transpose transpose_A) {
    // TODO(robertsdionne): implement.
  }

  template <>
  void VexClDevice<double>::Gemv(
      double alpha, const Values<double> &A, const Values<double> &x,
      double beta, Values<double> *y, Transpose transpose_A) {
    // TODO(robertsdionne): implement.
  }

  template <typename F>
  void VexClDevice<F>::Max(F alpha, const Values<F> &x, Values<F> *y) {
    y->values_device = max(alpha, x.values_device);
  }

  template <typename F>
  void VexClDevice<F>::MaxDerivative(
      F alpha, const Values<F> &dx, const Values<F> &y, Values<F> *dy) {
    dy->values_device = dx.values_device * (y.values_device > alpha);
  }

  template <typename F>
  void VexClDevice<F>::Softmax(const Values<F> &x, Values<F> *y) {
    auto do_max = vex::Reductor<F, vex::MAX>{context};
    auto do_sum = vex::Reductor<F, vex::SUM>{context};
    auto maximum = vex::make_temp<1>(do_max(x.values_device));
    auto exp_shifted = vex::make_temp<2>(exp(x.values_device - maximum));
    auto sum = vex::make_temp<3>(std::numeric_limits<F>::epsilon() + do_sum(exp_shifted));
    y->values_device = (std::numeric_limits<F>::epsilon() + exp_shifted) / sum;
  }

  template <typename F>
  void VexClDevice<F>::Initialize(Blob<F> &blob) {
    Initialize(blob.values);
    Initialize(blob.differences);
  }

  template <typename F>
  void VexClDevice<F>::Initialize(Values<F> &values) {
    assert(0 == values.values_device.size());
    if (0 == values.values_device.size()) {
      values.values_device = vex::vector<F>{context, values.values.size()};
    }
  }

  template <typename F>
  void VexClDevice<F>::Retrieve(Blob<F> &blob) {
    Retrieve(blob.values);
    Retrieve(blob.differences);
  }

  template <typename F>
  void VexClDevice<F>::Retrieve(Values<F> &values) {
    assert(values.values_device.size());
    if (values.values_device.size()) {
      vex::copy(
          values.values_device.begin(), values.values_device.end(), std::begin(values.values));
    }
  }

  template <typename F>
  void VexClDevice<F>::Ship(Blob<F> &blob) {
    Ship(blob.values);
    Ship(blob.differences);
  }

  template <typename F>
  void VexClDevice<F>::Ship(Values<F> &values) {
    assert(values.values_device.size());
    if (values.values_device.size()) {
      vex::copy(std::begin(values.values), std::end(values.values), values.values_device.begin());
    }
  }

  template class VexClDevice<float>;
  template class VexClDevice<double>;

}  // namespace autoencoder
