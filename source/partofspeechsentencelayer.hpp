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
        Blob &classify_weights, Blob &classify_bias,
        Blob &combine_weights, Blob &combine_bias,
        std::mt19937 &generator);

    virtual ~PartOfSpeechSentenceLayer() = default;

    void ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;

  private:
    float p;
    std::mt19937 &generator;
    Blob &classify_weights, &classify_bias;
    Blob &combine_weights, &combine_bias;
    std::vector<PartOfSpeechWordLayer> layers;
    std::vector<Blob> recurrent_states;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
