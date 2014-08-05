#version 440 core

layout(binding = 0) uniform sampler2D tex;

in vec2 texture_coordinate;

out vec4 fragment_color;

void main() {
  fragment_color = texture(tex, texture_coordinate);
}
