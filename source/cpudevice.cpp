#include AUTOENCODER_BLAS_HEADER
#include <limits>

#include "cpudevice.hpp"
#include "values.hpp"

namespace autoencoder {

  template <>
  void CpuDevice<float>::Axpby(float alpha, const Values<float> &x, float beta, Values<float> *y) {
    // TODO(robertsdionne): Figure out why clang thinks cblas_saxpby is an undefined symbol.
    cblas_sscal(y->width, beta, &y->values[0], 1);
    cblas_saxpy(x.width, alpha, &x.values[0], 1, &y->values[0], 1);
  }

  template <>
  void CpuDevice<double>::Axpby(
      double alpha, const Values<double> &x, double beta, Values<double> *y) {
    // TODO(robertsdionne): Figure out why clang thinks cblas_saxpby is an undefined symbol.
    cblas_dscal(y->width, beta, &y->values[0], 1);
    cblas_daxpy(x.width, alpha, &x.values[0], 1, &y->values[0], 1);
  }

  template <>
  void CpuDevice<float>::Gemm(
      float alpha, const Values<float> &A, const Values<float> &B, float beta, Values<float> *C,
      Transpose transpose_A, Transpose transpose_B) {
    cblas_sgemm(
        CblasColMajor, ToCblas(transpose_A), ToCblas(transpose_B),
        transpose_A == Transpose::kNo ? A.height : A.width,
        transpose_B == Transpose::kNo ? B.width : B.height,
        transpose_A == Transpose::kNo ? A.width : A.height,
        alpha, &A.values[0],
        A.height,
        &B.values[0],
        B.height,
        beta, &C->values[0], C->height);
  }

  template <>
  void CpuDevice<double>::Gemm(
      double alpha, const Values<double> &A, const Values<double> &B, double beta, Values<double> *C,
      Transpose transpose_A, Transpose transpose_B) {
    cblas_dgemm(
        CblasColMajor, ToCblas(transpose_A), ToCblas(transpose_B),
        transpose_A == Transpose::kNo ? A.height : A.width,
        transpose_B == Transpose::kNo ? B.width : B.height,
        transpose_A == Transpose::kNo ? A.width : A.height,
        alpha, &A.values[0],
        A.height,
        &B.values[0],
        B.height,
        beta, &C->values[0], C->height);
  }

  template <>
  void CpuDevice<float>::Gemv(
      float alpha, const Values<float> &A, const Values<float> &x, float beta, Values<float> *y,
      Transpose transpose_A) {
    cblas_sgemv(
        CblasColMajor, ToCblas(transpose_A),
        A.height, A.width,
        alpha, &A.values[0], A.height, &x.values[0], 1, beta, &y->values[0], 1);
  }

  template <>
  void CpuDevice<double>::Gemv(
      double alpha, const Values<double> &A, const Values<double> &x, double beta, Values<double> *y,
      Transpose transpose_A) {
    cblas_dgemv(
        CblasColMajor, ToCblas(transpose_A),
        A.height, A.width,
        alpha, &A.values[0], A.height, &x.values[0], 1, beta, &y->values[0], 1);
  }

  template <typename F>
  void CpuDevice<F>::Max(F alpha, const Values<F> &x, Values<F> *y) {
    y->values = x.values;
    y->values[y->values < alpha] = alpha;
  }

  template <typename F>
  void CpuDevice<F>::MaxDerivative(
      F alpha, const Values<F> &dx, const Values<F> &y, Values<F> *dy) {
    dy->values = dx.values * (y.values > alpha);
  }

  template <typename F>
  void CpuDevice<F>::Softmax(const Values<F> &x, Values<F> *y) {
    auto maximum = x.values.max();
    auto shifted = x.values - maximum;
    auto sum = std::numeric_limits<F>::epsilon() + std::exp(shifted).sum();
    y->values = (std::numeric_limits<F>::epsilon() + std::exp(shifted)) / sum;
  }

  template class CpuDevice<float>;
  template class CpuDevice<double>;

}  // namespace autoencoder
