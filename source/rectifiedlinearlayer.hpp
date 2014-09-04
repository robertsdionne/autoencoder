#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include "layer.hpp"
#include "values.hpp"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual ~RectifiedLinearLayer() = default;

    void ForwardCpu(const std::vector<Values *> &bottom, std::vector<Values *> *top) override;

    void BackwardCpu(const std::vector<Values *> &top, std::vector<Values *> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
