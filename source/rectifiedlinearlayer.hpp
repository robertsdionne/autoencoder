#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include <CL/cl.h>

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F> class Device;

  template <typename F>
  class RectifiedLinearLayer : public Layer<F> {
  public:
    RectifiedLinearLayer(Device<F> &device);

    virtual ~RectifiedLinearLayer();

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    F ForwardGpu(Mode mode, const Blobs<F> & bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;

  private:
    Device<F> &device;
    cl_program program = 0;
    cl_kernel kernel = 0;

    static constexpr const char *kSource = R"openclc(
      __kernel void RectifiedLinearForward(__global float *bottom, __global float *top) {
        int i = get_global_id(0);
        top[i] = max(0.0f, bottom[i]);
      }
    )openclc";
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
