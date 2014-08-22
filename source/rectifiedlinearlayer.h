#ifndef AUTOENCODER_RECTIFIEDLINEARLAYER_H_
#define AUTOENCODER_RECTIFIEDLINEARLAYER_H_

#include "layer.h"

namespace autoencoder {

  class RectifiedLinearLayer : public Layer {
  public:
    virtual void ForwardCpu(const std::vector<float *> &bottom, std::vector<float *> *top);

    virtual void BackwardCpu(const std::vector<float *> &top, std::vector<float *> *bottom);
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_RECTIFIEDLINEARLAYER_H_
