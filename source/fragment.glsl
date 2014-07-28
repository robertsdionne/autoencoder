#version 440 core

layout(binding = 0) uniform sampler1D tex;

out vec4 fragment_color;

int key(ivec2 v) {
  return ((v.y & 0x3ff) << 10) + (v.x & 0x3ff);
}

int h0(int k) {
  return (k * 2654435769 & 0xffffffff) >> (32 - 10);
}

int h1(int k) {
  return (k * 3667205999 & 0xffffffff) >> (32 - 10);
}

int h2(int k) {
  return (k * 903125161 & 0xffffffff) >> (32 - 10);
}

int h3(int k) {
  return (k * 2594636487 & 0xffffffff) >> (32 - 10);
}

int h4(int k) {
  return (k * 2458739852 & 0xffffffff) >> (32 - 10);
}

int h5(int k) {
  return (k * 4248052 & 0xffffffff) >> (32 - 10);
}

void main() {
  ivec2 index = ivec2(gl_FragCoord.xy) % 1024;
  if (texture(tex, h0(key(index)) / 1024.0).r > 0 &&
      texture(tex, h1(key(index)) / 1024.0).r > 0 &&
      texture(tex, h2(key(index)) / 1024.0).r > 0 &&
      texture(tex, h3(key(index)) / 1024.0).r > 0 &&
      texture(tex, h4(key(index)) / 1024.0).r > 0 &&
      texture(tex, h5(key(index)) / 1024.0).r > 0) {
    fragment_color = vec4(0.0f);
  } else {
    fragment_color = vec4(1.0f);
  }
}
