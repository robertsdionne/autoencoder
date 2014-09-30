#include <cassert>
#include <vexcl/vexcl.hpp>

#include "blob.hpp"
#include "values.hpp"
#include "vexcldevice.hpp"

namespace autoencoder {

  template <typename F>
  VexClDevice<F>::VexClDevice() : context{
    vex::Filter::Type{CL_DEVICE_TYPE_GPU} && vex::Filter::DoublePrecision
  } {
    assert(clblasSuccess == clblasSetup());
  }

  template <typename F>
  VexClDevice<F>::~VexClDevice() {
    clblasTeardown();
  }

  template <>
  void VexClDevice<float>::Axpby(
      float alpha, const Values<float> &x, float beta, Values<float> *y) {
    auto x_mem = x.values_device().raw();
    auto y_mem = y->values_device().raw();
    auto queue = context.queue(0)();
    cl_event complete;
    assert(clblasSuccess == clblasSscal(
        y->width, beta, y_mem, 0, 1,
        1, &queue,
        0, nullptr, nullptr));
    assert(clblasSuccess == clblasSaxpy(
        x.width, alpha, x_mem, 0, 1,
        y_mem, 0, 1,
        1, &queue,
        0, nullptr, nullptr));
  }

  template <>
  void VexClDevice<double>::Axpby(
      double alpha, const Values<double> &x, double beta, Values<double> *y) {
    auto x_mem = x.values_device().raw();
    auto y_mem = y->values_device().raw();
    auto queue = context.queue(0)();
    cl_event complete;
    assert(clblasSuccess == clblasDscal(
        y->width, beta, y_mem, 0, 1,
        1, &queue,
        0, nullptr, nullptr));
    assert(clblasSuccess == clblasDaxpy(
        x.width, alpha, x_mem, 0, 1,
        y_mem, 0, 1,
        1, &queue,
        0, nullptr, nullptr));
  }

  template <>
  void VexClDevice<float>::Gemm(
      float alpha, const Values<float> &A, const Values<float> &B, float beta, Values<float> *C,
      Transpose transpose_A, Transpose transpose_B) {
    auto A_mem = A.values_device().raw();
    auto B_mem = B.values_device().raw();
    auto C_mem = C->values_device().raw();
    auto queue = context.queue(0)();
    cl_event complete;
    assert(clblasSuccess == clblasSgemm(
        clblasColumnMajor, ToClBlas(transpose_A), ToClBlas(transpose_B),
        transpose_A == Transpose::kNo ? A.height : A.width,
        transpose_B == Transpose::kNo ? B.width : B.height,
        transpose_A == Transpose::kNo ? A.width : A.height,
        alpha, A_mem, 0, A.height,
        B_mem, 0, B.height,
        beta, C_mem, 0, C->height,
        1, &queue,
        0, nullptr, nullptr));
  }

  template <>
  void VexClDevice<double>::Gemm(
      double alpha, const Values<double> &A, const Values<double> &B,
      double beta, Values<double> *C, Transpose transpose_A, Transpose transpose_B) {
    auto A_mem = A.values_device().raw();
    auto B_mem = B.values_device().raw();
    auto C_mem = C->values_device().raw();
    auto queue = context.queue(0)();
    cl_event complete;
    assert(clblasSuccess == clblasDgemm(
        clblasColumnMajor, ToClBlas(transpose_A), ToClBlas(transpose_B),
        transpose_A == Transpose::kNo ? A.height : A.width,
        transpose_B == Transpose::kNo ? B.width : B.height,
        transpose_A == Transpose::kNo ? A.width : A.height,
        alpha, A_mem, 0, A.height,
        B_mem, 0, B.height,
        beta, C_mem, 0, C->height,
        1, &queue,
        0, nullptr, nullptr));
  }

  template <>
  void VexClDevice<float>::Gemv(
      float alpha, const Values<float> &A, const Values<float> &x, float beta, Values<float> *y,
      Transpose transpose_A) {
    auto A_mem = A.values_device().raw();
    auto x_mem = x.values_device().raw();
    auto y_mem = y->values_device().raw();
    auto queue = context.queue(0)();
    cl_event complete;
    assert(clblasSuccess == clblasSgemv(
        clblasColumnMajor, ToClBlas(transpose_A),
        A.height, A.width,
        alpha, A_mem, 0, A.height,
        x_mem, 0, 1,
        beta, y_mem, 0, 1,
        1, &queue,
        0, nullptr, nullptr));
  }

  template <>
  void VexClDevice<double>::Gemv(
      double alpha, const Values<double> &A, const Values<double> &x,
      double beta, Values<double> *y, Transpose transpose_A) {
    auto A_mem = A.values_device().raw();
    auto x_mem = x.values_device().raw();
    auto y_mem = y->values_device().raw();
    auto queue = context.queue(0)();
    cl_event complete;
    assert(clblasSuccess == clblasDgemv(
        clblasColumnMajor, ToClBlas(transpose_A),
        A.height, A.width,
        alpha, A_mem, 0, A.height,
        x_mem, 0, 1,
        beta, y_mem, 0, 1,
        1, &queue,
        0, nullptr, nullptr));
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

  template <>
  void VexClDevice<float>::SoftmaxDerivative(
      const Values<float> &x, const Values<float> &dx, Values<float> *dy) {
    VEX_FUNCTION(float, softmax_derivative, (size_t, n)(size_t, i)(float *, x)(float *, dx),
      float sum = 0.0f;
      float x_i = x[i];
      for (size_t j = 0; j < n; ++j) {
        sum += dx[j] * x[j] * ((i == j) - x_i);
      }
      return sum;
    );
    dy->values_device = softmax_derivative(x.values_device.size(), vex::element_index(),
        vex::raw_pointer(x.values_device), vex::raw_pointer(dx.values_device));
  }

  template <>
  void VexClDevice<double>::SoftmaxDerivative(
      const Values<double> &x, const Values<double> &dx, Values<double> *dy) {
    VEX_FUNCTION(
        double, softmax_derivative, (size_t, n)(size_t, i)(double *, x)(double *, dx),
      double sum = 0.0;
      double x_i = x[i];
      for (size_t j = 0; j < n; ++j) {
        sum += dx[j] * x[j] * ((i == j) - x_i);
      }
      return sum;
    );
    dy->values_device = softmax_derivative(x.values_device.size(), vex::element_index(),
        vex::raw_pointer(x.values_device), vex::raw_pointer(dx.values_device));
  }

  template <typename F>
  void VexClDevice<F>::Square(F alpha, const Values<F> &x, Values<F> *y) {
    y->values_device = alpha * vex::tag<1>(x.values_device) * vex::tag<1>(x.values_device);
  }

  template <typename F>
  F VexClDevice<F>::Sum(const Values<F> &x) {
    auto do_sum = vex::Reductor<F, vex::SUM>{context};
    return do_sum(x.values_device);
  }

  template <typename F>
  void VexClDevice<F>::Copy(const Values<F> &x, Values<F> *y) {
    y->values_device = x.values_device;
  }

  template <typename F>
  void VexClDevice<F>::Bernoulli(std::mt19937 &generator, F p, Values<F> *y) {
    vex::Random<F, vex::random::threefry> rng;
    y->values_device = rng(vex::element_index(), generator()) < p;
  }

  template <typename F>
  void VexClDevice<F>::Multiply(F alpha, const Values<F> &x, const Values<F> &y, Values<F> *z) {
    z->values_device = alpha * x.values_device * y.values_device;
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
