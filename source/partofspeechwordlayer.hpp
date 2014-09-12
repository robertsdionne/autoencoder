#ifndef AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
#define AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_

#include <random>

#include "blob.hpp"
#include "concatenatelayer.hpp"
#include "dropoutlayer.hpp"
#include "innerproductlayer.hpp"
#include "layer.hpp"
#include "rectifiedlinearlayer.hpp"
#include "softmaxlayer.hpp"

namespace autoencoder {

  class PartOfSpeechWordLayer : public Layer {
  public:
    PartOfSpeechWordLayer(
        float p,
        Blob &classify_weights, Blob &classify_bias,
        Blob &combine_weights, Blob &combine_bias,
        std::mt19937 &generator);

    virtual ~PartOfSpeechWordLayer() = default;

    float ForwardCpu(Mode mode, const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;

  private:
    DropoutLayer dropout;
    Blob corrupted_recurrent, corrupted_word;
    InnerProductLayer classify;
    Blob classified;
    SoftmaxLayer softmax;
    ConcatenateLayer concatenate;
    Blob concatenated;
    InnerProductLayer combine;
    Blob combined;
    RectifiedLinearLayer rectified_linear;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
