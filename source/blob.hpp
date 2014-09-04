#ifndef AUTOENCODER_PARAMETERS_HPP_
#define AUTOENCODER_PARAMETERS_HPP_

#include <vector>

#include "values.hpp"

namespace autoencoder {

  struct Blob {
    Blob(int width, int height = 1, int depth = 1, int duration = 1)
    : values(width, height, depth, duration), differences(width, height, depth, duration),
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

    void Reshape(int width, int height = 1, int depth = 1, int duration = 1) {
      width = width;
      height = height;
      depth = depth;
      duration = duration;
      values.Reshape(width, height, depth, duration);
      differences.Reshape(width, height, depth, duration);
    }

  public:
    Values values, differences;
    int width, height, depth, duration;
  };

  using Blobs = std::vector<Blob *>;

}  // namespace autoencoder

#endif  // AUTOENCODER_PARAMETERS_HPP_