#ifndef AUTOENCODER_MATRIX_H_
#define AUTOENCODER_MATRIX_H_

#include <cassert>
#include <iomanip>
#include <iostream>

#include "vector.h"

namespace autoencoder {

  struct Matrix {
    Matrix(int height, int width) : height(height), width(width) {
      values = new float[height * width]();
    }

    ~Matrix() {
      delete [] values;
    }

    float operator ()(int i, int j) const {
      assert(0 <= i && i < height && 0 <= j && j < width);
      return values[i * width + j];
    }

    float &operator ()(int i, int j) {
      assert(0 <= i && i < height && 0 <= j && j < width);
      return values[i * width + j];
    }

    Matrix operator +(const Matrix &other) {
      assert(other.height == height && other.width == width);
      auto result = Matrix(height, width);
      for (auto i = 0; i < result.height; ++i) {
        for (auto j = 0; j < result.width; ++j) {
          result(i, j) = this->operator()(i, j) + other(i, j);
        }
      }
      return result;
    }

    Vector operator *(const Vector &vector) {
      assert(vector.width == width);
      auto result = Vector(height);
      for (auto i = 0; i < height; ++i) {
        for (auto j = 0; j < width; ++j) {
          result(i) += this->operator()(i, j) * vector(j);
        }
      }
      return result;
    }

    float *values;
    int height, width;
  };

  static std::ostream &operator <<(std::ostream &out, const Matrix &matrix) {
    out << std::scientific << std::setprecision(2);
    for (auto i = 0; i < matrix.height; ++i) {
      for(auto j = 0; j < matrix.width; ++j) {
        out << std::setw(10) << matrix(i, j);
      }
      out << std::endl;
    }
    return out;
  }

}  // namespace autoencoder

#endif  // AUTOENCODER_MATRIX_H_
