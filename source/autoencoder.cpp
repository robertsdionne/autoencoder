#include <cassert>
#include <cmath>

#include "autoencoder.h"
#include "vector.h"

namespace autoencoder {

  Autoencoder::Autoencoder(int hidden, int input)
  : hidden(hidden), input(input), weights(hidden, input), bias(hidden) {}

  float Autoencoder::Activation(float in) {
    return tanh(in);
  }

  Vector Autoencoder::Activation(const Vector &in) {
    Vector out(in.width);
    for (auto i = 0; i < in.width; ++i) {
      out(i) = Activation(in(i));
    }
    return out;
  }

  Vector Autoencoder::Forward(const Vector &in) {
    assert(weights.width == in.width);
    Vector intermediate(weights.height);
    for (int i = 0; i < weights.height; ++i) {
      intermediate(i) = bias(i);
      for (int k = 0; k < weights.width; ++k) {
        intermediate(i) += weights(i, k) * in(k);
      }
    }
    return Activation(intermediate);
  }

}  // namespace autoencoder
