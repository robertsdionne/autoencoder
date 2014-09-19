#ifndef AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
#define AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_

#include <random>
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "partofspeechwordlayer.hpp"

namespace autoencoder {

  class PartOfSpeechSentenceLayer : public Layer {
  public:
    PartOfSpeechSentenceLayer(
        float p,
        Blob<float> &classify_weights, Blob<float> &classify_bias,
        Blob<float> &combine_weights, Blob<float> &combine_bias,
        std::mt19937 &generator);

    virtual ~PartOfSpeechSentenceLayer() = default;

    float ForwardCpu(Mode mode, const Blobs<float> &bottom, Blobs<float> *top) override;

    void BackwardCpu(const Blobs<float> &top, Blobs<float> *bottom) override;

  private:
    float p;
    std::mt19937 &generator;
    Blob<float> &classify_weights, &classify_bias;
    Blob<float> &combine_weights, &combine_bias;
    std::vector<PartOfSpeechWordLayer> layers;
    std::vector<Blob<float>> recurrent_states;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
