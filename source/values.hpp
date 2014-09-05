#ifndef AUTOENCODER_VALUES_HPP_
#define AUTOENCODER_VALUES_HPP_

#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

namespace autoencoder {

  struct Values {
    Values(int width = 1, int height = 1, int depth = 1, int duration = 1)
    : width(width), height(height), depth(depth), duration(duration),
      values(width * height * depth * duration) {}

    inline int Offset(int i, int j = 0, int k = 0, int l = 0) const {
      assert(0 <= i && i < width);
      assert(0 <= j && j < height);
      assert(0 <= k && k < depth);
      assert(0 <= l && l < duration);
      return ((i * height + j) * depth + k) * duration + l;
    }

    float value(int i, int j = 0, int k = 0, int l = 0) const {
      return values.at(Offset(i, j, k, l));
    }

    float &value(int i, int j = 0, int k = 0, int l = 0) {
      return values.at(Offset(i, j, k, l));
    }

    void Reshape(int width, int height = 1, int depth = 1, int duration = 1) {
      this->width = width;
      this->height = height;
      this->depth = depth;
      this->duration = duration;
      values.clear();
      values.resize(width * height * depth * duration);
    }

  public:
    std::vector<float> values;
    int width, height, depth, duration;
  };

  static std::ostream &operator <<(std::ostream &out, const Values &vector) {
    out << std::scientific << std::setprecision(2);
    for (auto i = 0; i < vector.width; ++i) {
      out << std::setw(10) << vector.value(i);
    }
    out << std::endl;
    return out;
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_VALUES_HPP_
