#ifndef AUTOENCODER_DROPOUTLAYER_HPP_
#define AUTOENCODER_DROPOUTLAYER_HPP_

#include <random>
#include <vector>

#include "blob.hpp"
#include "layer.hpp"

namespace autoencoder {

  template <typename F>
  class DropoutLayer : public Layer<F> {
  public:
    DropoutLayer(F p, std::mt19937 &generator);

    virtual ~DropoutLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;

  private:
    std::vector<Values<F>> mask;
    F p, scale;
    std::mt19937 &generator;
    std::bernoulli_distribution bernoulli;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_DROPOUTLAYER_HPP_
