#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_H_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_H_

#include "layer.h"
#include "values.h"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual ~RectifiedLinearLayer() = default;

    virtual void ForwardCpu(const Values &bottom, Values *top);

    virtual void BackwardCpu(const Values &top, Values *bottom);
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_H_
