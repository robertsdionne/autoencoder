#ifndef AUTOENCODER_VEXCLDEVICE_HPP_
#define AUTOENCODER_VEXCLDEVICE_HPP_

#include "device.hpp"

namespace autoencoder {

  template <typename F> class Blob;
  template <typename F> class Values;

  template <typename F>
  class VexClDevice : public Device<F> {
  public:
    VexClDevice();

    virtual ~VexClDevice() = default;

    virtual void Axpby(F alpha, const Values<F> &x, F beta, Values<F> *y);

    virtual void Gemm(F alpha, const Values<F> &A, const Values<F> &B, F beta, Values<F> *C,
        Transpose transpose_A = Transpose::kNo, Transpose transpose_B = Transpose::kNo);

    virtual void Gemv(F alpha, const Values<F> &A, const Values<F> &x, F beta, Values<F> *y,
        Transpose transpose_A = Transpose::kNo);

    void Retrieve(Blob<F> &blob);

    void Retrieve(Values<F> &values);

    void Ship(Blob<F> &blob);

    void Ship(Values<F> &values);

  private:
    vex::Context context;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_VEXCLDEVICE_HPP_