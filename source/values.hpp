#ifndef AUTOENCODER_VALUES_HPP_
#define AUTOENCODER_VALUES_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace autoencoder {

  template <typename F>
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

    F value(int i, int j = 0, int k = 0, int l = 0) const {
      return values.at(Offset(i, j, k, l));
    }

    F &value(int i, int j = 0, int k = 0, int l = 0) {
      return values.at(Offset(i, j, k, l));
    }

    inline int Argmax() const {
      auto maximum = -std::numeric_limits<F>::infinity();
      auto argmax = 0;
      for (auto i = 0; i < width; ++i) {
        if (maximum < value(i)) {
          maximum = value(i);
          argmax = i;
        }
      }
      return argmax;
    }

    inline bool IsFinite() const {
      auto result = true;
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              result &= std::isfinite(value(i, j, k, l));
            }
          }
        }
      }
      return result;
    }

    inline void IsValid() const {
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              assert(std::isfinite(value(i, j, k, l)));
            }
          }
        }
      }
    }

    void Reshape(int width, int height = 1, int depth = 1, int duration = 1) {
      this->width = width;
      this->height = height;
      this->depth = depth;
      this->duration = duration;
      Reset();
    }

    void Reset() {
      values.clear();
      values.resize(width * height * depth * duration);
    }

    F SquareMagnitude() {
      auto sum = F(0.0);
      for (auto i = 0; i < width; ++i) {
        for (auto j = 0; j < height; ++j) {
          for (auto k = 0; k < depth; ++k) {
            for (auto l = 0; l < duration; ++l) {
              sum += value(i, j, k, l) * value(i, j, k, l);
            }
          }
        }
      }
      return sum;
    }

  public:
    std::vector<F> values;
    int width, height, depth, duration;
  };

  template <typename F>
  static std::ostream &operator <<(std::ostream &out, const Values<F> &vector) {
    auto precision = out.precision();
    auto width = out.width();
    out << std::scientific << std::setprecision(2);
    for (auto i = 0; i < vector.width; ++i) {
      for (auto j = 0; j < vector.height; ++j) {
        for (auto k = 0; k < vector.depth; ++k) {
          for (auto l = 0; l < vector.duration; ++l) {
            out << std::setw(10) << vector.value(i, j, k, l);
          }
        }
      }
    }
    out << std::endl << std::fixed << std::setprecision(precision) << std::setw(width);
    return out;
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_VALUES_HPP_
