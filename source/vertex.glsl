#version 440 core

in vec4 vertex_position;

in vec2 tex_coordinate;

out vec2 texture_coordinate;

void main() {
  gl_Position = vertex_position;
  texture_coordinate = tex_coordinate;
}
