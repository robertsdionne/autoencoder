#ifndef AUTOENCODER_LAYER_HPP_
#define AUTOENCODER_LAYER_HPP_

#include <vector>

namespace autoencoder {

  struct Values;

  class Layer {
  public:
    virtual ~Layer() = default;

    inline void Forward(const std::vector<Values *> &bottom, std::vector<Values *> *top) {
      ForwardCpu(bottom, top);
    }

    inline void Backward(const std::vector<Values *> &top, std::vector<Values *> *bottom) {
      BackwardCpu(top, bottom);
    }

  protected:
    virtual void ForwardCpu(const std::vector<Values *> &bottom, std::vector<Values *> *top) = 0;

    virtual void ForwardGpu(const std::vector<Values *> &bottom, std::vector<Values *> *top) {
      ForwardCpu(bottom, top);
    }

    virtual void BackwardCpu(const std::vector<Values *> &top, std::vector<Values *> *bottom) = 0;

    virtual void BackwardGpu(const std::vector<Values *> &top, std::vector<Values *> *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_HPP_
