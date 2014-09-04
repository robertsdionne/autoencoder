#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include "layer.hpp"

namespace autoencoder {

  struct Parameters;

  class RectifiedLinearLayer : public Layer {
  public:
    virtual ~RectifiedLinearLayer() = default;

    void ForwardCpu(
        const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) override;

    void BackwardCpu(
        const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) override;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
