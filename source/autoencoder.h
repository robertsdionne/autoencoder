#ifndef AUTOENCODER_AUTOENCODER_H_
#define AUTOENCODER_AUTOENCODER_H_

#include "matrix.h"
#include "vector.h"

namespace autoencoder {

  class Autoencoder {
  public:
    Autoencoder(int hidden, int input);

    virtual ~Autoencoder() = default;

    float Activation(float in);

    Vector Activation(const Vector &in);

    Vector Forward(const Vector &in);

    int hidden, input;
    Matrix weights;
    Vector bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_AUTOENCODER_H_
