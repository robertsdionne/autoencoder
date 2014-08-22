#ifndef AUTOENCODER_VECTOR_H_
#define AUTOENCODER_VECTOR_H_

#include <cassert>
#include <iomanip>
#include <iostream>

namespace autoencoder {

  struct Vector {
    Vector(int width) : width(width) {
      values = new float[width]();
    }

    ~Vector() {
      delete [] values;
    }

    float operator ()(int i) const {
      assert(0 <= i && i < width);
      return values[i];
    }

    float &operator ()(int i) {
      assert(0 <= i && i < width);
      return values[i];
    }

    Vector operator +(const Vector &other) {
      assert(other.width == width);
      auto result = Vector(width);
      for (auto i = 0; i < result.width; ++i) {
        result(i) = this->operator()(i) + other(i);
      }
      return result;
    }

    float *values;
    int width;
  };

  static std::ostream &operator <<(std::ostream &out, const Vector &vector) {
    out << std::scientific << std::setprecision(2);
    for (auto i = 0; i < vector.width; ++i) {
      out << std::setw(10) << vector(i);
    }
    std::cout << std::endl;
    return out;
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_VECTOR_H_
