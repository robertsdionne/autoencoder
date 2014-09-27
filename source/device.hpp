#ifndef AUTOENCODER_DEVICE_HPP_
#define AUTOENCODER_DEVICE_HPP_

#include BLAS_HEADER

#include "interface.hpp"

namespace autoencoder {

  template <typename F> struct Values;

  enum class Transpose {
    kNo,
    kYes
  };

  template <typename F>
  class Device {
    DECLARE_INTERFACE(Device);

  public:
    virtual void Axpby(F alpha, const Values<F> &x, F beta, Values<F> *y) = 0;

    virtual void Gemm(F alpha, const Values<F> &A, const Values<F> &B, F beta, Values<F> *C,
        Transpose transpose_A = Transpose::kNo, Transpose transpose_B = Transpose::kNo) = 0;

    virtual void Gemv(F alpha, const Values<F> &A, const Values<F> &x, F beta, Values<F> *y,
        Transpose transpose_A = Transpose::kNo) = 0;
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

#endif  // AUTOENCODER_DEVICE_H_
