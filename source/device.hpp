#ifndef AUTOENCODER_DEVICE_HPP_
#define AUTOENCODER_DEVICE_HPP_

#include <random>

#include "interface.hpp"

namespace autoencoder {

  template <typename F> struct Blob;
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

    virtual void Max(F alpha, const Values<F> &x, Values<F> *y) = 0;

    virtual void MaxDerivative(F alpha, const Values<F> &dx, const Values<F> &y, Values<F> *dy) = 0;

    virtual void Softmax(const Values<F> &x, Values<F> *y) = 0;

    virtual void SoftmaxDerivative(const Values<F> &x, const Values<F> &dx, Values<F> *dy) = 0;

    virtual void Square(F alpha, const Values<F> &x, Values<F> *y) = 0;

    virtual F Sum(const Values<F> &x) = 0;

    virtual void Copy(const Values<F> &x, Values<F> *y) = 0;

    virtual void Bernoulli(std::mt19937 &generator, F p, Values<F> *y) = 0;

    virtual void Multiply(F alpha, const Values<F> &x, const Values<F> &y, Values<F> *z) = 0;

    virtual void Concatenate(const Values<F> &x, int offset, Values<F> *y) = 0;

    virtual void Split(int offset, const Values<F> &x, Values<F> *y) = 0;

    virtual void Initialize(Blob<F> &blob) = 0;

    virtual void Initialize(Values<F> &values) = 0;

    virtual void Retrieve(Blob<F> &blob) = 0;

    virtual void Retrieve(Values<F> &values) = 0;

    virtual void Ship(Blob<F> &blob) = 0;

    virtual void Ship(Values<F> &values) = 0;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DEVICE_H_
