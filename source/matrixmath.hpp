#ifndef AUTOENCODER_MATRIXMATH_HPP_
#define AUTOENCODER_MATRIXMATH_HPP_

#include BLAS_HEADER

namespace autoencoder {

  template <typename F>
  struct Values;

  void Saxpby(float alpha, const Values<float> &x, float beta, Values<float> *y);

  void Sgemm(
    float alpha, const Values<float> &A, const Values<float> &B, float beta, Values<float> *C,
    CBLAS_TRANSPOSE transpose_A = CblasNoTrans, CBLAS_TRANSPOSE transpose_B = CblasNoTrans);

  void Sgemv(
    float alpha, const Values<float> &A, const Values<float> &x, float beta, Values<float> *y,
    CBLAS_TRANSPOSE transpose_A = CblasNoTrans);

}  // namespace autoencoder

#endif  // AUTOENCODER_MATRIXMATH_HPP_
