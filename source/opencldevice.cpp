#include AUTOENCODER_OPENCL_HEADER
#include <cassert>
#include <clBLAS.h>
#include <gflags/gflags.h>

#include "blob.hpp"
#include "opencldevice.hpp"
#include "values.hpp"

namespace autoencoder {

#ifdef __APPLE__
  DEFINE_string(opencl_device_name, "GeForce", "The OpenCL device name");
#else
  DEFINE_string(opencl_device_name, "Cayman", "The OpenCL device name");
#endif

  template <typename F>
  OpenClDevice<F>::OpenClDevice() {

    // Get platform handle
    assert(CL_SUCCESS == clGetPlatformIDs(1, &platform, nullptr));

    // Get device handle
    cl_device_id devices[10] = {};
    cl_uint number_of_devices;
    assert(CL_SUCCESS == clGetDeviceIDs(
        platform, CL_DEVICE_TYPE_GPU, 10, devices, &number_of_devices));
    char buffer[1024] = {};
    size_t size = 0;
    for (auto i = 0; i < number_of_devices; ++i) {
      assert(CL_SUCCESS == clGetDeviceInfo(
          devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, &size));
      auto device_name = std::string{buffer};
      if (std::string::npos != device_name.find(FLAGS_opencl_device_name)) {
        device = devices[i];
      }
    }
    assert(device > 0);

    // Create context
    cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0
    };
    cl_int error;
    context = clCreateContext(properties, 1, &device, nullptr, nullptr, &error);
    assert(CL_SUCCESS == error);

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    assert(CL_SUCCESS == error);

    // Setup clBLAS
    assert(CL_SUCCESS == clblasSetup());
  }

  template <typename F>
  OpenClDevice<F>::~OpenClDevice() {

    // Teardown clBLAS
    clblasTeardown();

    // Release command queue
    if (queue) {
      assert(CL_SUCCESS == clReleaseCommandQueue(queue));
    }

    // Release context
    if (context) {
      assert(CL_SUCCESS == clReleaseContext(context));
    }
  }

  template <>
  void OpenClDevice<float>::Axpby(float alpha, const Values<float> &x, float beta, Values<float> *y) {
    // TODO(robertsdionne): implement
  }

  template <>
  void OpenClDevice<double>::Axpby(
      double alpha, const Values<double> &x, double beta, Values<double> *y) {
    // TODO(robertsdionne): implement
  }

  template <>
  void OpenClDevice<float>::Gemm(
      float alpha, const Values<float> &A, const Values<float> &B, float beta, Values<float> *C,
      Transpose transpose_A, Transpose transpose_B) {
    // TODO(robertsdionne): implement
  }

  template <>
  void OpenClDevice<double>::Gemm(
      double alpha, const Values<double> &A, const Values<double> &B, double beta, Values<double> *C,
      Transpose transpose_A, Transpose transpose_B) {
    // TODO(robertsdionne): implement
  }

  template <>
  void OpenClDevice<float>::Gemv(
      float alpha, const Values<float> &A, const Values<float> &x, float beta, Values<float> *y,
      Transpose transpose_A) {
    // TODO(robertsdionne): implement
  }

  template <>
  void OpenClDevice<double>::Gemv(
      double alpha, const Values<double> &A, const Values<double> &x, double beta, Values<double> *y,
      Transpose transpose_A) {
    // TODO(robertsdionne): implement
  }

  template <typename F>
  void OpenClDevice<F>::Retrieve(Blob<F> &blob) {
    Retrieve(blob.values);
    Retrieve(blob.differences);
  }

  template <typename F>
  void OpenClDevice<F>::Retrieve(Values<F> &values) {
    assert(values.memory);
    assert(CL_SUCCESS == clEnqueueReadBuffer(queue, values.memory, CL_TRUE, 0,
        values.width * values.height * values.depth * values.duration * sizeof(F),
        values.values.data(), 0, nullptr, nullptr));
  }

  template <typename F>
  void OpenClDevice<F>::Ship(Blob<F> &blob) {
    Ship(blob.values);
    Ship(blob.differences);
  }

  template <typename F>
  void OpenClDevice<F>::Ship(Values<F> &values) {
    if (!values.memory) {
      cl_int error;
      values.memory = clCreateBuffer(context, CL_MEM_READ_WRITE,
          values.width * values.height * values.depth * values.duration * sizeof(F),
          nullptr, &error);
      assert(CL_SUCCESS == error);
    }
    assert(CL_SUCCESS == clEnqueueWriteBuffer(queue, values.memory, CL_TRUE, 0,
        values.width * values.height * values.depth * values.duration * sizeof(F),
        values.values.data(), 0, nullptr, nullptr));
  }

  template class OpenClDevice<float>;
  template class OpenClDevice<double>;

}  // namespace autoencoder
