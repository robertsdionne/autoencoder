#include <cblas.h>

#include "matrixmath.h"
#include "values.h"

namespace autoencoder {

  void Sgemm(float alpha, const Values &A, const Values &B, float beta, Values *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0f, A.values, 1,
        B.values, 1, 0.0f, C->values, 1);
    for (auto i = 0; i < A.height; ++i) {
      for (auto j = 0; j < B.width; ++j) {
        C->value(j, i) *= beta;
        for (auto k = 0; k < A.width; ++k) {
          C->value(j, i) += alpha * B.value(j, k) * A.value(k, i);
        }
      }
    }
  }

}  // namespace autoencoder
