#include BLAS_HEADER

#include "matrixmath.hpp"
#include "values.hpp"

namespace autoencoder {

  void Sgemm(float alpha, const Values &A, const Values &B, float beta, Values *C) {
    cblas_sgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        A.height, B.width, A.width,
        alpha, A.values, A.height, B.values, B.height, beta, C->values, C->height);
  }

  void Sgemv(float alpha, const Values &A, const Values &x, float beta, Values *y) {
    cblas_sgemv(
        CblasColMajor, CblasNoTrans,
        A.height, A.width,
        alpha, A.values, A.height, x.values, 1, beta, y->values, 1);
  }

}  // namespace autoencoder
