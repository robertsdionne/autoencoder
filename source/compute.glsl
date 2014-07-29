#version 440 core

layout(local_size_x = 32, local_size_y = 32) in;

layout(rgba32f, location = 0) uniform image2D fragments;
uniform float time;
uniform vec2 mouse;

const int kMaxLoops = 100;
const float kEpsilon = 1e-3;

float Cube(vec3 position);
float DDistance(vec3 position, vec3 d);
float Distance(vec3 position);
vec3 Normal(vec3 position);
vec3 Ray(uvec3 index);
float Sphere(vec3 position);
void Store(vec4 color);
vec4 Trace(vec3 ray);

void main() {
  Store(Trace(Ray(gl_GlobalInvocationID)));
}

float Cube(vec3 position) {
  vec3 distance = abs(position) - vec3(0.5, 0.01, 0.5);
  return min(max(max(distance.x, distance.y), distance.z),
      length(max(distance, 0.0)));
}

float DDistance(vec3 position, vec3 d) {
  return Distance(position + d) - Distance(position - d);
}

float Distance(vec3 position) {
  return Cube(position+vec3(0,0.05,0)*sin(15.0*position.x+25.0*time) + vec3(0.0, 0.1, 1.0));
}

vec3 Normal(vec3 position) {
  const vec3 dx = vec3(kEpsilon, 0.0, 0.0);
  const vec3 dy = vec3(0.0, kEpsilon, 0.0);
  const vec3 dz = vec3(0.0, 0.0, kEpsilon);
  return normalize(vec3(DDistance(position, dx), DDistance(position, dy), DDistance(position, dz)));
}

vec3 Ray(uvec3 index) {
  vec3 dimensions = vec3(gl_NumWorkGroups * gl_WorkGroupSize);
  vec2 center = dimensions.xy / 2.0;
  vec3 coordinate = vec3(gl_GlobalInvocationID.xy / 1024.0, -1.0) - vec3(0.5, 0.5, 0.0);
  return normalize(coordinate);
}

float Sphere(vec3 position) {
  return length(position) - sqrt(0.5);
}

void Store(vec4 color) {
  imageStore(fragments, ivec2(gl_GlobalInvocationID.xy), color);
}

vec4 Trace(vec3 ray) {
  vec3 position = vec3(0.0);
  bool hit = false;
  bool was_hit = false;
  float count = 0.0;
  for (int i = 0; i < kMaxLoops; ++i) {
    float distance = Distance(position);
    position += ray * distance;
    hit = distance < kEpsilon;
    was_hit = was_hit || hit;
    count += float(!was_hit && position.z > -1000.0) / float(kMaxLoops);
  }
  const vec4 color = vec4(1.0);
  return float(hit) * color * (max(dot(normalize(vec3(-10.0,0.0,0.0)-position), Normal(position)), 0.0) + vec4(vec3(0.0), 0.0));
  // return vec4(vec3(count), 1.0);
}
