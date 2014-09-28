#include <vexcl/vexcl.hpp>

#include "blob.hpp"
#include "values.hpp"
#include "vexcldevice.hpp"

namespace autoencoder {

  template <typename F>
  VexClDevice<F>::VexClDevice() : context{vex::Filter::Type{CL_DEVICE_TYPE_GPU}} {}

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
  void VexClDevice<F>::Retrieve(Blob<F> &blob) {
    Retrieve(blob.values);
    Retrieve(blob.differences);
  }

  template <typename F>
  void VexClDevice<F>::Retrieve(Values<F> &values) {
    vex::copy(values.values_device, values.values);
  }

  template <typename F>
  void VexClDevice<F>::Ship(Blob<F> &blob) {
    Ship(blob.values);
    Ship(blob.differences);
  }

  template <typename F>
  void VexClDevice<F>::Ship(Values<F> &values) {
    vex::copy(values.values, values.values_device);
  }

}  // namespace autoencoder