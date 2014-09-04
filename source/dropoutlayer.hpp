#ifndef AUTOENCODER_DROPOUTLAYER_HPP_
#define AUTOENCODER_DROPOUTLAYER_HPP_

#include <gflags/gflags.h>
#include <random>

#include "blob.hpp"
#include "layer.hpp"

DECLARE_int32(random_seed);

namespace autoencoder {

  class DropoutLayer : public Layer {
  public:
    DropoutLayer(float p, unsigned int random_seed = FLAGS_random_seed);

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
