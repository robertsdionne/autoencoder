#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_

#include "layer.hpp"
#include "values.hpp"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual ~RectifiedLinearLayer() = default;

    virtual void ForwardCpu(const Values &bottom, Values *top);

    virtual void BackwardCpu(const Values &top, Values *bottom);
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_HPP_
