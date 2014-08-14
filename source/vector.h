#ifndef AUTOENCODER_VECTOR_H_
#define AUTOENCODER_VECTOR_H_

namespace autoencoder {

  struct Vector {
    Vector(int width) : width(width) {
      values = new float[width]();
    }

    ~Vector() {
      delete [] values;
    }

    float &operator ()(int i) const {
      assert(0 <= i && i < width);
      return values[i];
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
