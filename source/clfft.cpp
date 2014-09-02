#include <cassert>
#include <clFFT.h>
#include <cmath>
#include <gflags/gflags.h>
#include <iostream>
#include <string>

constexpr size_t kN = 16;

DEFINE_string(target_gpu, "GeForce", "The OpenCL GPU device name");

int main(int argument_count, char *arguments[]) {
  gflags::SetUsageMessage("clFFT demo program.");
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);

  cl_platform_id platform = 0;
  assert(CL_SUCCESS == clGetPlatformIDs(1, &platform, nullptr));
  cl_device_id device = 0;
  cl_device_id devices[10] = {};
  cl_uint number_of_devices;
  assert(CL_SUCCESS == clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_GPU, 10, devices, &number_of_devices));

  char buffer[1024] = {};
  size_t size = 0;
  for (auto i = 0; i < number_of_devices; ++i) {
    assert(CL_SUCCESS == clGetDeviceInfo(
        devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, &size));
    std::string device_name = buffer;
    if (std::string::npos != device_name.find(FLAGS_target_gpu)) {
      device = devices[i];
    }
  }
  assert(device > 0);

  cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0
  };
  cl_int error;
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
  assert(CL_SUCCESS == error);

  clfftSetupData fft_setup;
  assert(CL_SUCCESS == clfftInitSetupData(&fft_setup));
  assert(CL_SUCCESS == clfftSetup(&fft_setup));

  float *x = new float[2 * kN]();

  for (auto i = 0; i < 2 * kN; ++i) {
    x[i] = i / 2 + 1;
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;

  cl_mem buffer_x = clCreateBuffer(
      context, CL_MEM_READ_WRITE, 2 * kN * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);

  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_x, CL_TRUE,  0, 2 * kN * sizeof(float), x, 0, nullptr, nullptr));

  clfftPlanHandle plan_handle;
  clfftDim dimension = CLFFT_1D;
  size_t lengths[1] = {kN};
  assert(CL_SUCCESS == clfftCreateDefaultPlan(&plan_handle, context, dimension, lengths));

  assert(CL_SUCCESS == clfftSetPlanPrecision(plan_handle, CLFFT_SINGLE));
  assert(CL_SUCCESS == clfftSetLayout(
      plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
  assert(CL_SUCCESS == clfftSetResultLocation(plan_handle, CLFFT_INPLACE));

  assert(CL_SUCCESS == clfftBakePlan(plan_handle, 1, &queue, nullptr, nullptr));

  assert(CL_SUCCESS == clfftEnqueueTransform(
      plan_handle, CLFFT_FORWARD, 1, &queue, 0, nullptr, nullptr, &buffer_x, nullptr, nullptr));

  assert(CL_SUCCESS == clFinish(queue));

  assert(CL_SUCCESS == clEnqueueReadBuffer(
      queue, buffer_x, CL_TRUE, 0, 2 * kN * sizeof(float), x, 0, nullptr, nullptr));

  for (auto i = 0; i < 2 * kN; ++i) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;

  clReleaseMemObject(buffer_x);

  delete [] x;

  assert(CL_SUCCESS == clfftDestroyPlan(&plan_handle));

  clfftTeardown();

  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
