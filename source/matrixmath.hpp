#ifndef AUTOENCODER_MATRIXMATH_HPP_
#define AUTOENCODER_MATRIXMATH_HPP_

#include BLAS_HEADER

namespace autoencoder {

  template <typename F>
  struct Values;

  template <typename F>
  void Axpby(F alpha, const Values<F> &x, F beta, Values<F> *y);

  template <typename F>
  void Gemm(
    F alpha, const Values<F> &A, const Values<F> &B, F beta, Values<F> *C,
    CBLAS_TRANSPOSE transpose_A = CblasNoTrans, CBLAS_TRANSPOSE transpose_B = CblasNoTrans);

  template <typename F>
  void Gemv(
    F alpha, const Values<F> &A, const Values<F> &x, F beta, Values<F> *y,
    CBLAS_TRANSPOSE transpose_A = CblasNoTrans);

}  // namespace autoencoder

#endif  // AUTOENCODER_MATRIXMATH_HPP_
