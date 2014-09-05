#ifndef AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
#define AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_

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
        unsigned int random_seed = FLAGS_random_seed);

    virtual ~PartOfSpeechWordLayer() = default;

    void ForwardCpu(const Blobs &bottom, Blobs *top) override;

    void BackwardCpu(const Blobs &top, Blobs *bottom) override;

  private:
    ConcatenateLayer concatenate;
    Blob concatenated;
    DropoutLayer dropout;
    Blob corrupted;
    InnerProductLayer classify, combine;
    Blob classified, combined;
    SoftmaxLayer softmax;
    RectifiedLinearLayer rectified_linear;
  };

}  // namespace autoencoder

#endif  // AUTOENCODER_PARTOFSPEECHWORDLAYER_HPP_
