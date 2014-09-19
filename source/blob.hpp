#ifndef AUTOENCODER_PARAMETERS_HPP_
#define AUTOENCODER_PARAMETERS_HPP_

#include <limits>
#include <vector>

#include "values.hpp"

namespace autoencoder {

  struct Blob {
    Blob(int width = 1, int height = 1, int depth = 1, int duration = 1)
    : values(width, height, depth, duration), differences(width, height, depth, duration),
      velocities(width, height, depth, duration), accelerations(width, height, depth, duration),
      width(width), height(height), depth(depth), duration(duration) {}

    float difference(int i, int j = 0, int k = 0, int l = 0) const {
      return differences.value(i, j, k, l);
    }

    float &difference(int i, int j = 0, int k = 0, int l = 0) {
      return differences.value(i, j, k, l);
    }

    float value(int i, int j = 0, int k = 0, int l = 0) const {
      return values.value(i, j, k, l);
    }

    float &value(int i, int j = 0, int k = 0, int l = 0) {
      return values.value(i, j, k, l);
    }

    float velocity(int i, int j = 0, int k = 0, int l = 0) const {
      return velocities.value(i, j, k, l);
    }

    float &velocity(int i, int j = 0, int k = 0, int l = 0) {
      return velocities.value(i, j, k, l);
    }

    float acceleration(int i, int j = 0, int k = 0, int l = 0) const {
      return accelerations.value(i, j, k, l);
    }

    float &acceleration(int i, int j = 0, int k = 0, int l = 0) {
      return accelerations.value(i, j, k, l);
    }

    inline bool IsFinite() const {
      return values.IsFinite() && differences.IsFinite();
    }

    inline void IsValid() const {
      values.IsValid();
      differences.IsValid();
    }

    void Reshape(int width, int height = 1, int depth = 1, int duration = 1) {
      this->width = width;
      this->height = height;
      this->depth = depth;
      this->duration = duration;
      values.Reshape(width, height, depth, duration);
      differences.Reshape(width, height, depth, duration);
      velocities.Reshape(width, height, depth, duration);
      accelerations.Reshape(width, height, depth, duration);
    }

    float SquareMagnitude() {
      auto sum = 0.0f;
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              sum += difference(i, j, k, l) * difference(i, j, k, l);
            }
          }
        }
      }
      return sum;
    }

    void ClipGradient(float sum) {
      auto magnitude = sqrt(sum);
      if (magnitude > 1.0) {
        for (auto i = 0; i < width; ++i) {
          for (auto j = 0; j < height; ++j) {
            for (auto k = 0; k < depth; ++k) {
              for (auto l = 0; l < duration; ++l) {
                difference(i, j, k, l) /= magnitude;
              }
            }
          }
        }
      }
    }

    inline void UpdateMomentum(float learning_rate, float momentum = 0.0f) {
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              velocity(i, j, k, l) =
                  momentum * velocity(i, j, k, l) - learning_rate * difference(i, j, k, l);
              value(i, j, k, l) += velocity(i, j, k, l);
            }
          }
        }
      }
    }

    inline void UpdateAdaDelta(float learning_rate, float decay) {
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              velocity(i, j, k, l) = decay * velocity(i, j, k, l)
                  + (1.0 - decay) * difference(i, j, k, l) * difference(i, j, k, l);
              auto delta = -sqrt(acceleration(i, j, k, l) + std::numeric_limits<float>::epsilon())
                  / sqrt(velocity(i, j, k, l) + std::numeric_limits<float>::epsilon())
                  * difference(i, j, k, l);
              acceleration(i, j, k, l) = decay * acceleration(i, j, k, l)
                  + (1.0 - decay) * delta * delta;
              value(i, j, k, l) += learning_rate * delta;
            }
          }
        }
      }
      IsValid();
    }

    inline void UpdateAdaGrad(float learning_rate) {
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              velocity(i, j, k, l) += difference(i, j, k, l) * difference(i, j, k, l);
              value(i, j, k, l) -= learning_rate *
                  difference(i, j, k, l) / (sqrt(velocity(i, j, k, l) + 1.0));
            }
          }
        }
      }
    }

  public:
    Values values, differences, velocities, accelerations;
    int width, height, depth, duration;
  };

  using Blobs = std::vector<Blob *>;

}  // namespace autoencoder

#endif  // AUTOENCODER_PARAMETERS_HPP_
