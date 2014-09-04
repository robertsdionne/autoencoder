#ifndef AUTOENCODER_MATRIXMATH_HPP_
#define AUTOENCODER_MATRIXMATH_HPP_

#include BLAS_HEADER

namespace autoencoder {

  struct Values;

  void Saxpby(float alpha, const Values &x, float beta, Values *y);

  void Sgemm(
    float alpha, const Values &A, const Values &B, float beta, Values *C,
    CBLAS_TRANSPOSE transpose_A = CblasNoTrans, CBLAS_TRANSPOSE transpose_B = CblasNoTrans);

  void Sgemv(
    float alpha, const Values &A, const Values &x, float beta, Values *y,
    CBLAS_TRANSPOSE transpose_A = CblasNoTrans);

}  // namespace autoencoder

#endif  // AUTOENCODER_MATRIXMATH_HPP_
