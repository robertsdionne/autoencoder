#ifndef AUTOENCODER_OPENCLDEVICE_HPP_
#define AUTOENCODER_OPENCLDEVICE_HPP_

#include <clBLAS.h>
#include <gflags/gflags.h>

#include "device.hpp"

namespace autoencoder {

  DECLARE_string(opencl_device_name);

  template <typename F>
  class OpenClDevice : public Device<F> {
  public:
    OpenClDevice();

    virtual ~OpenClDevice();

    virtual void Axpby(F alpha, const Values<F> &x, F beta, Values<F> *y);

    virtual void Gemm(F alpha, const Values<F> &A, const Values<F> &B, F beta, Values<F> *C,
        Transpose transpose_A = Transpose::kNo, Transpose transpose_B = Transpose::kNo);

    virtual void Gemv(F alpha, const Values<F> &A, const Values<F> &x, F beta, Values<F> *y,
        Transpose transpose_A = Transpose::kNo);

  private:
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context context = 0;
    cl_command_queue queue = 0;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_OPENCLDEVICE_HPP_
