#include BLAS_HEADER

#include "matrixmath.h"
#include "values.h"

namespace autoencoder {

  void Sgemm(float alpha, const Values &A, const Values &B, float beta, Values *C) {
    cblas_sgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        A.height, B.width, A.width,
        1.0f, A.values, A.height, B.values, B.height, 0.0f, C->values, C->height);
  }

}  // namespace autoencoder
