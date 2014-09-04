#ifndef AUTOENCODER_INNERPRODUCTLAYER_HPP_
#define AUTOENCODER_INNERPRODUCTLAYER_HPP_

#include "layer.hpp"

namespace autoencoder {

  struct Parameters;

  class InnerProductLayer : public Layer {
  public:
    InnerProductLayer(Parameters &weights, Parameters &bias);

    virtual ~InnerProductLayer() = default;

    void ForwardCpu(
        const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) override;

    void BackwardCpu(
        const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) override;

  private:
    Parameters &weights;
    Parameters &bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_HPP_
