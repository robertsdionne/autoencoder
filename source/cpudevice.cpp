#include BLAS_HEADER

#include "cpudevice.hpp"
#include "values.hpp"

namespace autoencoder {

template <>
  void CpuDevice<float>::Axpby(float alpha, const Values<float> &x, float beta, Values<float> *y) {
    // TODO(robertsdionne): Figure out why clang thinks cblas_saxpby is an undefined symbol.
    cblas_sscal(y->width, beta, y->values.data(), 1);
    cblas_saxpy(x.width, alpha, x.values.data(), 1, y->values.data(), 1);
  }

  template <>
  void CpuDevice<double>::Axpby(
      double alpha, const Values<double> &x, double beta, Values<double> *y) {
    // TODO(robertsdionne): Figure out why clang thinks cblas_saxpby is an undefined symbol.
    cblas_dscal(y->width, beta, y->values.data(), 1);
    cblas_daxpy(x.width, alpha, x.values.data(), 1, y->values.data(), 1);
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
        alpha, A.values.data(),
        A.height,
        B.values.data(),
        B.height,
        beta, C->values.data(), C->height);
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
        alpha, A.values.data(),
        A.height,
        B.values.data(),
        B.height,
        beta, C->values.data(), C->height);
  }

  template <>
  void CpuDevice<float>::Gemv(
      float alpha, const Values<float> &A, const Values<float> &x, float beta, Values<float> *y,
      Transpose transpose_A) {
    cblas_sgemv(
        CblasColMajor, ToCblas(transpose_A),
        A.height, A.width,
        alpha, A.values.data(), A.height, x.values.data(), 1, beta, y->values.data(), 1);
  }

  template <>
  void CpuDevice<double>::Gemv(
      double alpha, const Values<double> &A, const Values<double> &x, double beta, Values<double> *y,
      Transpose transpose_A) {
    cblas_dgemv(
        CblasColMajor, ToCblas(transpose_A),
        A.height, A.width,
        alpha, A.values.data(), A.height, x.values.data(), 1, beta, y->values.data(), 1);
  }

}  // namespace autoencoder
