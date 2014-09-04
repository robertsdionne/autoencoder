#ifndef AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
#define AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_

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
        Blob &combine_weights, Blob &combine_bias);

    virtual ~PartOfSpeechSentenceLayer() = default;

    void ForwardCpu(const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;

  private:
    float p;
    Blob &classify_weights, &classify_bias;
    Blob &combine_weights, &combine_bias;
    std::vector<PartOfSpeechWordLayer> layers;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHSENTENCELAYER_HPP_
