#version 440 core

layout(binding = 0) uniform sampler1D tex;

out vec4 fragment_color;

void main() {
  int index = int(gl_FragCoord.x) % 1024;
  if (texture(tex, index / 1024.0f).r > 0) {
    fragment_color = vec4(1.0f);
  } else {
    fragment_color = vec4(0.0f);
  }
}
