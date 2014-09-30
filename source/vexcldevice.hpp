#ifndef AUTOENCODER_VEXCLDEVICE_HPP_
#define AUTOENCODER_VEXCLDEVICE_HPP_

#include <clBLAS.h>
#include <random>

#include "device.hpp"

namespace autoencoder {

  template <typename F> class Blob;
  template <typename F> class Values;

  template <typename F>
  class VexClDevice : public Device<F> {
  public:
    VexClDevice();

    virtual ~VexClDevice();

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

    void Copy(const Values<F> &x, Values<F> *y) override;

    void Bernoulli(std::mt19937 &generator, F p, Values<F> *y) override;

    void Multiply(F alpha, const Values<F> &x, const Values<F> &y, Values<F> *z) override;

    void Initialize(Blob<F> &blob) override;

    void Initialize(Values<F> &values) override;

    void Retrieve(Blob<F> &blob) override;

    void Retrieve(Values<F> &values) override;

    void Ship(Blob<F> &blob) override;

    void Ship(Values<F> &values) override;

  private:
    vex::Context context;
  };

  inline clblasTranspose ToClBlas(Transpose transpose) {
    switch (transpose) {
      case Transpose::kYes:
        return clblasTrans;
        break;
      case Transpose::kNo:
        return clblasNoTrans;
        break;
    }
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_VEXCLDEVICE_HPP_
