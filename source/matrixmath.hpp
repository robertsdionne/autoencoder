#ifndef AUTOENCODER_MATRIXMATH_H_
#define AUTOENCODER_MATRIXMATH_H_

namespace autoencoder {

  struct Values;

  void Sgemm(float alpha, const Values &A, const Values &B, float beta, Values *C);

}  // namespace autoencoder

#endif  // AUTOENCODER_MATRIXMATH_H_
