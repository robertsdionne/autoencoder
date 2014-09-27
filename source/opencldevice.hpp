#ifndef AUTOENCODER_OPENCLDEVICE_HPP_
#define AUTOENCODER_OPENCLDEVICE_HPP_

#include "device.hpp"

namespace autoencoder {

  template <typename F>
  class OpenClDevice : public Device<F> {
  public:
    virtual void Axpby(F alpha, const Values<F> &x, F beta, Values<F> *y);

    virtual void Gemm(F alpha, const Values<F> &A, const Values<F> &B, F beta, Values<F> *C,
        Transpose transpose_A = Transpose::kNo, Transpose transpose_B = Transpose::kNo);

    virtual void Gemv(F alpha, const Values<F> &A, const Values<F> &x, F beta, Values<F> *y,
        Transpose transpose_A = Transpose::kNo);
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_OPENCLDEVICE_HPP_
