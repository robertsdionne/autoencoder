#ifndef AUTOENCODER_LAYER_H_
#define AUTOENCODER_LAYER_H_

#include <vector>

namespace autoencoder {

  class Layer {
  public:
    virtual ~Layer() = default;

    inline void Forward(const std::vector<float *> &bottom, std::vector<float *> *top) {
      ForwardCpu(bottom, top);
    }

    inline void Backward(const std::vector<float *> &top, std::vector<float *> *bottom) {
      BackwardCpu(top, bottom);
    }

  protected:
    virtual void ForwardCpu(const std::vector<float *> &bottom, std::vector<float *> *top) = 0;

    virtual void ForwardGpu(const std::vector<float *> &bottom, std::vector<float *> *top) {
      ForwardCpu(bottom, top);
    }

    virtual void BackwardCpu(const std::vector<float *> &top, std::vector<float *> *bottom) = 0;

    virtual void BackwardGpu(const std::vector<float *> &top, std::vector<float *> *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_H_
