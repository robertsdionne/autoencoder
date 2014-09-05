#include BLAS_HEADER

#include "matrixmath.hpp"
#include "values.hpp"

namespace autoencoder {

  void Saxpby(float alpha, const Values &x, float beta, Values *y) {
    // TODO(robertsdionne): Figure out why clang thinks cblas_saxpby is an undefined symbol.
    cblas_sscal(y->width, beta, y->values.data(), 1);
    cblas_saxpy(x.width, alpha, x.values.data(), 1, y->values.data(), 1);
  }

  void Sgemm(
      float alpha, const Values &A, const Values &B, float beta, Values *C,
      CBLAS_TRANSPOSE transpose_A, CBLAS_TRANSPOSE transpose_B) {
    cblas_sgemm(
        CblasColMajor, transpose_A, transpose_B,
        transpose_A == CblasNoTrans ? A.height : A.width,
        transpose_B == CblasNoTrans ? B.width : B.height,
        transpose_A == CblasNoTrans ? A.width : A.height,
        alpha, A.values.data(),
        A.height,
        B.values.data(),
        B.height,
        beta, C->values.data(), C->height);
  }

  void Sgemv(
      float alpha, const Values &A, const Values &x, float beta, Values *y,
      CBLAS_TRANSPOSE transpose_A) {
    cblas_sgemv(
        CblasColMajor, transpose_A,
        A.height, A.width,
        alpha, A.values.data(), A.height, x.values.data(), 1, beta, y->values.data(), 1);
  }

}  // namespace autoencoder
