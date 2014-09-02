#include <cassert>
#include <clBLAS.h>
#include <cstdio>
#include <gflags/gflags.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/types.h>

constexpr int kM = 4;
constexpr int kN = 3;
constexpr int kK = 5;

constexpr float alpha = 10;

const float A[kM * kK] = {
  11, 12, 13, 14, 15,
  21, 22, 23, 24, 25,
  31, 32, 33, 34, 35,
  41, 42, 43, 44, 45
};
constexpr size_t kLda = kK;

const float B[kK * kN] = {
  11, 12, 13,
  21, 22, 23,
  31, 32, 33,
  41, 42, 43,
  51, 52, 53
};
constexpr size_t kLdb = kN;

constexpr float beta = 20;

static const float C[kM * kN] = {
  11, 12, 13,
  21, 22, 23,
  31, 32, 33,
  41, 42, 43
};
constexpr size_t kLdc = kN;

static float result[kM * kN];

DEFINE_string(target_gpu, "GeForce", "The OpenCL GPU device name");

int main(int argument_count, char *arguments[]) {
  gflags::SetUsageMessage("clBLAS demo program.");
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

  assert(CL_SUCCESS == clblasSetup());

  cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, kM * kK * sizeof(*A), nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, kK * kN * sizeof(*B), nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_mem buffer_c = clCreateBuffer(context, CL_MEM_READ_WRITE, kM * kN * sizeof(*C), nullptr, &error);
  assert(CL_SUCCESS == error);

  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_a, CL_TRUE, 0, kM * kK * sizeof(*A), A, 0, nullptr, nullptr));
  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_b, CL_TRUE, 0, kK * kN * sizeof(*B), B, 0, nullptr, nullptr));
  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_c, CL_TRUE, 0, kM * kN * sizeof(*C), C, 0, nullptr, nullptr));

  cl_event event;
  assert(CL_SUCCESS == clblasSgemm(
      clblasRowMajor, clblasNoTrans, clblasNoTrans,
      kM, kN, kK,
      alpha, buffer_a, 0, kLda,
      buffer_b, 0, kLdb, beta,
      buffer_c, 0, kLdc,
      1, &queue, 0, nullptr, &event));

  assert(CL_SUCCESS == clWaitForEvents(1, &event));

  assert(CL_SUCCESS == clEnqueueReadBuffer(
      queue, buffer_c, CL_TRUE, 0, kM * kN * sizeof(*result), result, 0, nullptr, nullptr));

  assert(CL_SUCCESS == clReleaseMemObject(buffer_a));
  assert(CL_SUCCESS == clReleaseMemObject(buffer_b));
  assert(CL_SUCCESS == clReleaseMemObject(buffer_c));

  std::cout << std::scientific << std::setprecision(2);
  for (auto i = 0; i < kM; ++i) {
    for (auto j = 0; j < kN; ++j) {
      std::cout << std::setw(10) << result[i * kN + j];
    }
    std::cout << std::endl;
  }

  clblasTeardown();

  assert(CL_SUCCESS == clReleaseCommandQueue(queue));
  assert(CL_SUCCESS == clReleaseContext(context));

  return 0;
}
