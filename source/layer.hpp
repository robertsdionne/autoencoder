#ifndef AUTOENCODER_LAYER_HPP_
#define AUTOENCODER_LAYER_HPP_

#include <vector>

namespace autoencoder {

  struct Parameters;

  class Layer {
  public:
    virtual ~Layer() = default;

    inline void Forward(const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) {
      ForwardCpu(bottom, top);
    }

    inline void Backward(const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) {
      BackwardCpu(top, bottom);
    }

  protected:
    virtual void ForwardCpu(
        const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) = 0;

    virtual void ForwardGpu(
        const std::vector<Parameters *> &bottom, std::vector<Parameters *> *top) {
      ForwardCpu(bottom, top);
    }

    virtual void BackwardCpu(
        const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) = 0;

    virtual void BackwardGpu(
        const std::vector<Parameters *> &top, std::vector<Parameters *> *bottom) {
      BackwardCpu(top, bottom);
    }
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_LAYER_HPP_
