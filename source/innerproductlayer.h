#ifndef AUTOENCODER_INNERPRODUCTLAYER_H_
#define AUTOENCODER_INNERPRODUCTLAYER_H_

#include "layer.h"
#include "values.h"

namespace autoencoder {

  class InnerProductLayer : public Layer {
  public:
    InnerProductLayer(const Values &weights, const Values &bias);

    virtual ~InnerProductLayer() = default;

    virtual void ForwardCpu(const Values &bottom, Values *top);

    virtual void BackwardCpu(const Values &top, Values *bottom);

  private:
    Values weights;
    Values bias;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_INNERPRODUCTLAYER_H_
