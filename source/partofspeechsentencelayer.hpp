#ifndef AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
#define AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_

#include <random>
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "partofspeechwordlayer.hpp"

namespace autoencoder {

  template <typename F> class Device;

  template <typename F>
  class PartOfSpeechSentenceLayer : public Layer<F> {
  public:
    PartOfSpeechSentenceLayer(
        Device<F> &device,
        F p,
        Blob<F> &classify_weights, Blob<F> &classify_bias,
        Blob<F> &combine_weights, Blob<F> &combine_bias,
        std::mt19937 &generator);

    virtual ~PartOfSpeechSentenceLayer() = default;

    F ForwardCpu(Mode mode, const Blobs<F> &bottom, Blobs<F> *top) override;

    void BackwardCpu(const Blobs<F> &top, Blobs<F> *bottom) override;

  private:
    Device<F> &device;
    F p;
    std::mt19937 &generator;
    Blob<F> &classify_weights, &classify_bias;
    Blob<F> &combine_weights, &combine_bias;
    std::vector<PartOfSpeechWordLayer<F>> layers;
    std::vector<Blob<F>> recurrent_states;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
