#ifndef AUTOENCODER_MATRIXMATH_HPP_
#define AUTOENCODER_MATRIXMATH_HPP_

namespace autoencoder {

  struct Values;

  void Sgemm(float alpha, const Values &A, const Values &B, float beta, Values *C);

  void Sgemv(float alpha, const Values &A, const Values &x, float beta, Values *y);

}  // namespace autoencoder

#endif  // AUTOENCODER_MATRIXMATH_HPP_
