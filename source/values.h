#ifndef AUTOENCODER_VALUES_H_
#define AUTOENCODER_VALUES_H_

#include <cassert>
#include <iomanip>
#include <iostream>

namespace autoencoder {

  struct Values {
    Values(int width) : width(width) {
      values = new float[width]();
      differences = new float[width]();
    }

    ~Values() {
      delete [] values;
      delete [] differences;
    }

    float difference(int i) const {
      assert(0 <= i && i < width);
      return differences[i];
    }

    float &difference(int i) {
      assert(0 <= i && i < width);
      return differences[i];
    }

    float value(int i) const {
      assert(0 <= i && i < width);
      return values[i];
    }

    float &value(int i) {
      assert(0 <= i && i < width);
      return values[i];
    }

    Values operator +(const Values &other) {
      assert(other.width == width);
      auto result = Values(width);
      for (auto i = 0; i < result.width; ++i) {
        result.value(i) = this->value(i) + other.value(i);
      }
      return result;
    }

    float *values, *differences;
    int width;
  };

  static std::ostream &operator <<(std::ostream &out, const Values &vector) {
    out << std::scientific << std::setprecision(2);
    for (auto i = 0; i < vector.width; ++i) {
      out << std::setw(10) << vector.value(i);
    }
    std::cout << std::endl;
    return out;
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_VALUES_H_
