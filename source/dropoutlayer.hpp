#ifndef AUTOENCODER_DROPOUTLAYER_HPP_
#define AUTOENCODER_DROPOUTLAYER_HPP_

#include <random>

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class DropoutLayer : public Layer {
  public:
    DropoutLayer(float p);

    virtual ~DropoutLayer() = default;

    void ForwardCpu(const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;

  private:
    Blob mask;
    float p, scale;
    std::mt19937 generator;
    std::bernoulli_distribution bernoulli;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DROPOUTLAYER_HPP_
