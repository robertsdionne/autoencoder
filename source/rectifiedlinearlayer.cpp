#include <cassert>

#include "blob.hpp"
#include "opencldevice.hpp"
#include "rectifiedlinearlayer.hpp"

namespace autoencoder {

  template <typename F>
  constexpr const char *RectifiedLinearLayer<F>::kSource;

  template <typename F>
  RectifiedLinearLayer<F>::RectifiedLinearLayer(Device<F> &device) : device(device) {}

  template <typename F>
  RectifiedLinearLayer<F>::~RectifiedLinearLayer() {
    if (program) {
      assert(CL_SUCCESS == clReleaseKernel(kernel));
      kernel = 0;
      assert(CL_SUCCESS == clReleaseProgram(program));
      program = 0; 
    }
  }

  template <typename F>
  F RectifiedLinearLayer<F>::ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    for (auto i = 0; i < top->at(0)->width; ++i) {
      top->at(0)->value(i) = std::max(F(0.0), bottom.at(0)->value(i));
    }
    top->at(0)->IsValid();
    return F(0.0);
  }

  template <typename F>
  F RectifiedLinearLayer<F>::ForwardGpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) {
    auto &opencl_device = dynamic_cast<OpenClDevice<F> &>(device);
    opencl_device.Ship(bottom.at(0)->values);
    if (!program) {
      cl_int error;
      program = clCreateProgramWithSource(
          opencl_device.context, 1, const_cast<const char **>(&kSource), nullptr, &error);
      assert(CL_SUCCESS == error);
      if (CL_SUCCESS != clBuildProgram(
          program, 1, &opencl_device.device, nullptr, nullptr, nullptr)) {
        char log[4096];
        assert(CL_SUCCESS == clGetProgramBuildInfo(
            program, opencl_device.device, CL_PROGRAM_BUILD_LOG, sizeof(log), &log, nullptr));
        std::cout << log << std::endl;
        assert(false);
      }
      kernel = clCreateKernel(program, "RectifiedLinearForward", &error);
      assert(CL_SUCCESS == error);
    }
    assert(CL_SUCCESS == clSetKernelArg(kernel, 0, sizeof(cl_mem), bottom.at(0)->values.memory));
    assert(CL_SUCCESS == clSetKernelArg(kernel, 1, sizeof(cl_mem), top->at(0)->values.memory));
    cl_event complete;
    auto global_work_size = size_t(bottom.at(0)->values.width);
    /*assert(CL_SUCCESS ==*/ std::cout << clEnqueueNDRangeKernel(opencl_device.queue,
        kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, &complete)/*)*/ << std::endl;
    assert(CL_SUCCESS == clWaitForEvents(1, &complete));
    return F(0.0);
  }

  template <typename F>
  void RectifiedLinearLayer<F>::BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) {
    for (auto i = 0; i < bottom->at(0)->width; ++i) {
      bottom->at(0)->difference(i) = top.at(0)->difference(i) * (bottom->at(0)->value(i) > F(0.0));
    }
    bottom->at(0)->IsValid();
  }

  template class RectifiedLinearLayer<float>;
  template class RectifiedLinearLayer<double>;

}  // namespace autoencoder
