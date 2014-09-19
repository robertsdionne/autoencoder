#ifndef AUTOENCODER_DROPOUTLAYER_HPP_
#define AUTOENCODER_DROPOUTLAYER_HPP_

#include <random>
#include <vector>

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  class DropoutLayer : public Layer {
  public:
    DropoutLayer(float p, std::mt19937 &generator);

    virtual ~DropoutLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;

  private:
    std::vector<Values<float>> mask;
    float p, scale;
    std::mt19937 &generator;
    std::bernoulli_distribution bernoulli;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DROPOUTLAYER_HPP_
