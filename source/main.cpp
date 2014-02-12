#include <OpenCL/cl.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "checks.h"

int main(int argument_count, const char *arguments[]) {
  std::vector<cl::Platform> platforms;
  CHECK_STATE(CL_SUCCESS == cl::Platform::get(&platforms));
  std::cout << platforms.size() << std::endl;
  auto platform = platforms[0];
  std::string extensions, name, profile, vendor, version;
  CHECK_STATE(CL_SUCCESS == platform.getInfo(CL_PLATFORM_EXTENSIONS, &extensions));
  CHECK_STATE(CL_SUCCESS == platform.getInfo(CL_PLATFORM_NAME, &name));
  CHECK_STATE(CL_SUCCESS == platform.getInfo(CL_PLATFORM_PROFILE, &profile));
  CHECK_STATE(CL_SUCCESS == platform.getInfo(CL_PLATFORM_VENDOR, &vendor));
  CHECK_STATE(CL_SUCCESS == platform.getInfo(CL_PLATFORM_VERSION, &version));
  std::cout << extensions << std::endl
    << name << std::endl
    << profile << std::endl
    << vendor << std::endl
    << version << std::endl;
  
  return 0;
}
