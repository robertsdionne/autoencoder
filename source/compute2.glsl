#version 440 core

layout(local_size_x = 2, local_size_y = 1, local_size_z = 2) in;

// m # of weights
// d dimension of samples
// n # of samples
layout(r32f, location = 0) readonly uniform image2D weights; // m x d matrix
layout(r32f, location = 1) readonly uniform image2D samples; // d x n matrix
layout(r32f, location = 2) volatile uniform image2D result; // m x n matrix

void main() {
  ivec2 ij = ivec2(gl_GlobalInvocationID.xz);
  vec4 value = vec4(0.0);
  for (uint k = 0; k < gl_WorkGroupSize.x; ++k) {
    ivec2 ik = ivec2(ij.x, gl_WorkGroupSize.x * gl_WorkGroupID.y + k);
    ivec2 kj = ivec2(gl_WorkGroupSize.x * gl_WorkGroupID.y + k, ij.y);
    value += imageLoad(weights, ik) * imageLoad(samples, kj);
  }
  imageStore(result, ij, value);
}
