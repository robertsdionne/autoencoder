#ifndef AUTOENCODER_CPU_HPP_
#define AUTOENCODER_CPU_HPP_

#include AUTOENCODER_BLAS_HEADER

#include "device.hpp"

namespace autoencoder {

  template <typename F> struct Values;

  template <typename F>
  class CpuDevice : public Device<F> {
  public:
    void Axpby(F alpha, const Values<F> &x, F beta, Values<F> *y) override;

    void Gemm(F alpha, const Values<F> &A, const Values<F> &B, F beta, Values<F> *C,
        Transpose transpose_A = Transpose::kNo, Transpose transpose_B = Transpose::kNo) override;

    void Gemv(F alpha, const Values<F> &A, const Values<F> &x, F beta, Values<F> *y,
        Transpose transpose_A = Transpose::kNo) override;

    void Max(F alpha, const Values<F> &x, Values<F> *y) override;

    void MaxDerivative(F alpha, const Values<F> &dx, const Values<F> &y, Values<F> *dy) override;

    void Softmax(const Values<F> &x, Values<F> *y) override;

    void SoftmaxDerivative(const Values<F> &x, const Values<F> &dx, Values<F> *dy) override;

    void Square(F alpha, const Values<F> &x, Values<F> *y) override;

    F Sum(const Values<F> &x) override;
  };

  inline CBLAS_TRANSPOSE ToCblas(Transpose transpose) {
    switch (transpose) {
      case Transpose::kYes:
        return CblasTrans;
        break;
      case Transpose::kNo:
        return CblasNoTrans;
        break;
    }
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_CPU_HPP_
