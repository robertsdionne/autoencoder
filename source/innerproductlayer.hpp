#ifndef AUTOENCODER_INNERPRODUCTLAYER_HPP_
#define AUTOENCODER_INNERPRODUCTLAYER_HPP_

#include "layer.hpp"
#include "values.hpp"

namespace autoencoder {

  class InnerProductLayer : public Layer {
  public:
    InnerProductLayer(const Values &weights, const Values &bias);

    virtual ~InnerProductLayer() = default;

    void ForwardCpu(const std::vector<Values *> &bottom, std::vector<Values *> *top) override;

    void BackwardCpu(const std::vector<Values *> &top, std::vector<Values *> *bottom) override;

  private:
    Values weights;
    Values bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_HPP_
