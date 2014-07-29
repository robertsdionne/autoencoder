#version 440 core

layout(local_size_x = 32, local_size_y = 32) in;

layout(rgba32f, location = 0) uniform image2D fragments;

uniform float time;
uniform vec2 mouse;

void main() {
  ivec2 index = ivec2(gl_GlobalInvocationID.xy);
  vec4 color = vec4(gl_WorkGroupID.xyy / 32.0f, 1.0f);
  imageStore(fragments, index, color);
}
